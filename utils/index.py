import math
from datetime import date
from typing import Literal

import pendulum


def get_quarter(which: str | int = "current", bounds: Literal["start", "end"] = "start") -> date:
    """
    Return the beginning date or end date of today's quarter as a date object.
    """
    d = date.today()
    # beginning of current quarter
    month_begin = math.ceil(d.month / 3) * 3 - 2
    dt_begin = date(d.year, month_begin, 1)

    if which == "current":
        offset = 0
    elif which == "last":
        offset = -1
    else:
        try:
            offset = int(which)
        except Exception:
            raise ValueError(
                'Argument "which" needs to be one of "current", "last", or convertible to int.'
            )
    # add offset
    pend_begin = pendulum.parse(dt_begin.isoformat())
    pend_modified = pend_begin.add(months=3 * offset)  # type: ignore

    if bounds == "start":
        return date(pend_modified.year, pend_modified.month, pend_modified.day)
    elif bounds == "end":
        pend_end = pend_modified.add(months=3).subtract(days=1)
        return date(pend_end.year, pend_end.month, pend_end.day)
    else:
        raise ValueError(f'Argument "bounds" needs to be in ["start", "end"], got {bounds}.')
