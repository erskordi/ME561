from datetime import datetime, timezone
import numpy as np
from sgp4.api import Satrec, jday, SGP4_ERRORS


class SGP4Propagator:
    def __init__(self, tle_line1: str, tle_line2: str):
        self.sat = Satrec.twoline2rv(tle_line1, tle_line2)

    def propagate(self, dt_utc: datetime):
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        else:
            dt_utc = dt_utc.astimezone(timezone.utc)

        jd, fr = jday(
            dt_utc.year,
            dt_utc.month,
            dt_utc.day,
            dt_utc.hour,
            dt_utc.minute,
            dt_utc.second + dt_utc.microsecond * 1e-6
        )

        error_code, r_km, v_km_s = self.sat.sgp4(jd, fr)

        if error_code != 0:
            message = SGP4_ERRORS.get(error_code, "Unknown SGP4 error")
            raise RuntimeError(f"SGP4 failed with error {error_code}: {message}")

        r_km = np.array(r_km, dtype=np.float64)
        v_km_s = np.array(v_km_s, dtype=np.float64)

        return r_km, v_km_s