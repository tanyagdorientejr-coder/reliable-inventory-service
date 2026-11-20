import heapq
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import RLock
from typing import Callable, Dict, List, Optional, Tuple

DEFAULT_MAX_TTL_SECONDS = 30 * 24 * 3600   # 30 days
DEFAULT_RETENTION_SECONDS = 24 * 3600      # idempotency / history window
_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)


class InventoryError(ValueError):
    """Base class (subclasses ValueError so existing callers keep working)."""


class InvalidRequest(InventoryError):
    pass


class InvalidClock(InventoryError):
    """The injected clock returned something other than an aware datetime."""


class ProductNotFound(InventoryError):
    pass


class InsufficientInventory(InventoryError):
    pass


class ReservationNotFound(InventoryError):
    pass


class InvalidStateTransition(InventoryError):
    pass


class IdempotencyConflict(InventoryError):
    pass


class Status(str, Enum):
    ACTIVE = "ACTIVE"
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class Reservation:
    """Immutable snapshot. Callers can't mutate service state through it."""
    reservation_id: str
    customer_id: str
    product_id: str
    quantity: int
    idempotency_key: str
    ttl_seconds: int
    status: Status
    expires_at: datetime


def _check_int(value, name: str, minimum: int, maximum: Optional[int] = None):
    # bool is a subclass of int; reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InvalidRequest(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise InvalidRequest(f"{name} must be <= {maximum}")


def _check_str(value, name: str):
    if not isinstance(value, str) or not value.strip():
        raise InvalidRequest(f"{name} must be a non-empty string")


class InventoryService:
    """Thread-safe in-memory reservation engine.

    Invariant (per product), checked by `check_invariants`:
        available + held by ACTIVE reservations + committed (confirmed)
        == total stock ever added

    Retention: once a reservation reaches a terminal state (CONFIRMED,
    CANCELLED, EXPIRED) it, and its idempotency key, are kept for
    `retention_seconds` and then purged. Within that window a retried request
    with the same key gets the original result; after it, the key is reusable
    and the old reservation id is no longer found. retention_seconds must be a
    positive integer; 0 is rejected because it would purge in the same sweep and
    break idempotent retries. Pass None to keep everything (memory then grows
    without bound).
    """

    _COMPACT_MIN = 64  # minimum stale entries before the expiry heap is rebuilt

    def __init__(
        self,
        clock: Callable[[], datetime] = None,
        max_ttl_seconds: int = DEFAULT_MAX_TTL_SECONDS,
        retention_seconds: Optional[int] = DEFAULT_RETENTION_SECONDS,
    ):
        _check_int(max_ttl_seconds, "max_ttl_seconds", 1)
        if retention_seconds is not None:
            _check_int(retention_seconds, "retention_seconds", 1)
        self._max_ttl = max_ttl_seconds
        self._retention = retention_seconds
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._now()  # fail fast on a bad clock

        self._lock = RLock()
        self._available: Dict[str, int] = {}
        self._total_stock: Dict[str, int] = {}
        self._committed: Dict[str, int] = {}
        self._reservations: Dict[str, Reservation] = {}
        # (customer_id, idempotency_key) -> reservation_id
        self._idem: Dict[Tuple[str, str], str] = {}
        # min-heap of (expires_at, id); entries go stale when a reservation
        # leaves ACTIVE early, and are compacted once they dominate the heap.
        self._expiry_heap: List[Tuple[datetime, str]] = []
        self._stale = 0
        # min-heap of (purge_at, id), one entry per terminal reservation
        self._purge_heap: List[Tuple[datetime, str]] = []

    def _now(self) -> datetime:
        """Single entry point for time; guarantees a timezone-aware value so
        comparisons with stored expiry times can never raise TypeError."""
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise InvalidClock("clock must return a timezone-aware datetime")
        return now

    # ---- stock ---------------------------------------------------------

    def add_product(self, product_id: str, quantity: int) -> None:
        """Create a product or add stock to it. Never overwrites, so stock
        held by existing reservations can't be silently double-counted."""
        _check_str(product_id, "product_id")
        _check_int(quantity, "quantity", 0)
        with self._lock:
            self._available[product_id] = self._available.get(product_id, 0) + quantity
            self._total_stock[product_id] = self._total_stock.get(product_id, 0) + quantity

    def get_available_inventory(self, product_id: str) -> int:
        _check_str(product_id, "product_id")
        with self._lock:
            self._sweep_locked(self._now())
            return self._available.get(product_id, 0)

    # ---- reservations --------------------------------------------------

    def reserve(
        self,
        customer_id: str,
        product_id: str,
        quantity: int,
        idempotency_key: str,
        ttl_seconds: int = 900,
    ) -> Reservation:
        _check_str(customer_id, "customer_id")
        _check_str(product_id, "product_id")
        _check_str(idempotency_key, "idempotency_key")
        _check_int(quantity, "quantity", 1)
        _check_int(ttl_seconds, "ttl_seconds", 1, self._max_ttl)

        with self._lock:
            # Everything that can fail (clock, datetime arithmetic) happens
            # BEFORE any state is touched, so a failure can't leak stock.
            now = self._now()
            try:
                expires_at = now + timedelta(seconds=ttl_seconds)
            except OverflowError:
                raise InvalidRequest("ttl_seconds puts expiry beyond the supported date range")

            self._sweep_locked(now)

            key = (customer_id, idempotency_key)
            existing_id = self._idem.get(key)
            if existing_id is not None:
                existing = self._reservations[existing_id]
                if (existing.product_id, existing.quantity, existing.ttl_seconds) != (
                    product_id, quantity, ttl_seconds
                ):
                    raise IdempotencyConflict(
                        "idempotency_key was already used with different parameters"
                    )
                return existing

            if product_id not in self._available:
                raise ProductNotFound("Product does not exist")
            if self._available[product_id] < quantity:
                raise InsufficientInventory("Insufficient inventory")

            reservation = Reservation(
                reservation_id=str(uuid.uuid4()),
                customer_id=customer_id,
                product_id=product_id,
                quantity=quantity,
                idempotency_key=idempotency_key,
                ttl_seconds=ttl_seconds,
                status=Status.ACTIVE,
                expires_at=expires_at,
            )
            # Commit: only infallible mutations from here on.
            self._available[product_id] -= quantity
            self._reservations[reservation.reservation_id] = reservation
            self._idem[key] = reservation.reservation_id
            heapq.heappush(self._expiry_heap, (expires_at, reservation.reservation_id))
            return reservation

    def confirm(self, reservation_id: str) -> Reservation:
        _check_str(reservation_id, "reservation_id")
        with self._lock:
            now = self._now()
            self._sweep_locked(now)  # an overdue reservation must not be confirmable
            r = self._get_locked(reservation_id)
            if r.status is Status.CONFIRMED:
                return r  # idempotent
            if r.status is not Status.ACTIVE:
                raise InvalidStateTransition(f"Cannot confirm a {r.status.value} reservation")
            self._committed[r.product_id] = self._committed.get(r.product_id, 0) + r.quantity
            return self._leave_active_locked(r, Status.CONFIRMED, now)

    def cancel(self, reservation_id: str) -> Reservation:
        _check_str(reservation_id, "reservation_id")
        with self._lock:
            now = self._now()
            self._sweep_locked(now)
            r = self._get_locked(reservation_id)
            if r.status is Status.CANCELLED:
                return r  # idempotent
            if r.status is not Status.ACTIVE:
                # EXPIRED stock was already returned; CONFIRMED stock is sold.
                raise InvalidStateTransition(f"Cannot cancel a {r.status.value} reservation")
            self._available[r.product_id] += r.quantity
            return self._leave_active_locked(r, Status.CANCELLED, now)

    def expire_reservations(self) -> List[str]:
        with self._lock:
            return self._sweep_locked(self._now())

    def get_reservation(self, reservation_id: str) -> Reservation:
        _check_str(reservation_id, "reservation_id")
        with self._lock:
            self._sweep_locked(self._now())
            return self._get_locked(reservation_id)

    # ---- internals (caller holds the lock) -----------------------------

    def _get_locked(self, reservation_id: str) -> Reservation:
        r = self._reservations.get(reservation_id)
        if r is None:
            raise ReservationNotFound("Reservation not found")
        return r

    def _terminal_locked(self, r: Reservation, status: Status, now: datetime) -> Reservation:
        updated = replace(r, status=status)
        self._reservations[r.reservation_id] = updated
        if self._retention is not None:
            try:
                purge_at = now + timedelta(seconds=self._retention)
            except OverflowError:
                purge_at = _FAR_FUTURE
            heapq.heappush(self._purge_heap, (purge_at, r.reservation_id))
        return updated

    def _leave_active_locked(self, r: Reservation, status: Status, now: datetime) -> Reservation:
        self._stale += 1  # its expiry-heap entry is now dead weight
        return self._terminal_locked(r, status, now)

    def _sweep_locked(self, now: datetime) -> List[str]:
        """Expire overdue reservations, purge old terminal ones, compact."""
        expired: List[str] = []
        heap = self._expiry_heap
        while heap and heap[0][0] <= now:
            _, rid = heapq.heappop(heap)
            r = self._reservations.get(rid)
            if r is None or r.status is not Status.ACTIVE:
                self._stale = max(0, self._stale - 1)  # consumed a stale entry
                continue
            self._available[r.product_id] += r.quantity
            self._terminal_locked(r, Status.EXPIRED, now)
            expired.append(rid)

        purge = self._purge_heap
        while purge and purge[0][0] <= now:
            _, rid = heapq.heappop(purge)
            r = self._reservations.pop(rid, None)
            if r is not None:
                self._idem.pop((r.customer_id, r.idempotency_key), None)

        if self._stale > self._COMPACT_MIN and self._stale > len(heap) - self._stale:
            self._expiry_heap = [
                (r.expires_at, r.reservation_id)
                for r in self._reservations.values()
                if r.status is Status.ACTIVE
            ]
            heapq.heapify(self._expiry_heap)
            self._stale = 0
        return expired

    def check_invariants(self) -> None:
        with self._lock:
            held: Dict[str, int] = {}
            for r in self._reservations.values():
                if r.status is Status.ACTIVE:
                    held[r.product_id] = held.get(r.product_id, 0) + r.quantity
            for pid, total in self._total_stock.items():
                avail = self._available[pid]
                assert avail >= 0, f"{pid}: negative inventory"
                assert avail + held.get(pid, 0) + self._committed.get(pid, 0) == total, (
                    f"{pid}: inventory drift"
                )
            assert len(self._idem) == len(self._reservations), "idempotency index drift"


if __name__ == "__main__":
    service = InventoryService()
    service.add_product("GPU-RTX-5060", 10)

    first = service.reserve("customer-001", "GPU-RTX-5060", 3, "order-1001")
    second = service.reserve("customer-001", "GPU-RTX-5060", 3, "order-1001")

    print("First reservation:", first)
    print("Second reservation:", second)
    print("Same reservation:", first.reservation_id == second.reservation_id)
    print("Available inventory:", service.get_available_inventory("GPU-RTX-5060"))
