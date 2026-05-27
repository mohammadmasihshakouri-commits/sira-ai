def lookup_order(reservation_code):
    return {
        "found": True,
        "reservation_code": reservation_code,
        "status": "eligible_for_cancellation",
        "customer_phone": "09120000000",
        "message": "Order found and eligible for cancellation."
    }