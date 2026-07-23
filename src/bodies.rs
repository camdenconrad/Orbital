//! Static physical data for the major bodies.
//!
//! The catalog covers everything with meaningful gravitational influence on a
//! spacecraft: Sun, planets, Earth's Moon, the Galilean moons, Titan, Triton,
//! Pluto, and the four most massive main-belt asteroids (Ceres, Vesta, Pallas,
//! Hygiea — together ~half the belt's mass).

/// Speed of light, km/s.
pub const C_KM_S: f64 = 299_792.458;
pub const AU_KM: f64 = 149_597_870.7;
pub const DAY_S: f64 = 86_400.0;
pub const CENTURY_D: f64 = 36_525.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BodyId {
    Sun,
    Mercury,
    Venus,
    Earth,
    Moon,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
    Pluto,
    // Major moons (beyond Luna).
    Io,
    Europa,
    Ganymede,
    Callisto,
    Titan,
    Triton,
    // Most massive asteroids / dwarf planets of the belt.
    Ceres,
    Vesta,
    Pallas,
    Hygiea,
}

pub const ALL_BODIES: [BodyId; 21] = [
    BodyId::Sun,
    BodyId::Mercury,
    BodyId::Venus,
    BodyId::Earth,
    BodyId::Moon,
    BodyId::Mars,
    BodyId::Jupiter,
    BodyId::Saturn,
    BodyId::Uranus,
    BodyId::Neptune,
    BodyId::Pluto,
    BodyId::Io,
    BodyId::Europa,
    BodyId::Ganymede,
    BodyId::Callisto,
    BodyId::Titan,
    BodyId::Triton,
    BodyId::Ceres,
    BodyId::Vesta,
    BodyId::Pallas,
    BodyId::Hygiea,
];

impl BodyId {
    /// Index into [`ALL_BODIES`]. `BodyId` is fieldless and declared in the
    /// same order as the array, so the discriminant *is* the index — no scan.
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            BodyId::Sun => "Sun",
            BodyId::Mercury => "Mercury",
            BodyId::Venus => "Venus",
            BodyId::Earth => "Earth",
            BodyId::Moon => "Moon",
            BodyId::Mars => "Mars",
            BodyId::Jupiter => "Jupiter",
            BodyId::Saturn => "Saturn",
            BodyId::Uranus => "Uranus",
            BodyId::Neptune => "Neptune",
            BodyId::Pluto => "Pluto",
            BodyId::Io => "Io",
            BodyId::Europa => "Europa",
            BodyId::Ganymede => "Ganymede",
            BodyId::Callisto => "Callisto",
            BodyId::Titan => "Titan",
            BodyId::Triton => "Triton",
            BodyId::Ceres => "Ceres",
            BodyId::Vesta => "Vesta",
            BodyId::Pallas => "Pallas",
            BodyId::Hygiea => "Hygiea",
        }
    }

    /// Parent planet for moons; None for heliocentric bodies.
    pub fn parent(self) -> Option<BodyId> {
        match self {
            BodyId::Moon => Some(BodyId::Earth),
            BodyId::Io | BodyId::Europa | BodyId::Ganymede | BodyId::Callisto => {
                Some(BodyId::Jupiter)
            }
            BodyId::Titan => Some(BodyId::Saturn),
            BodyId::Triton => Some(BodyId::Neptune),
            _ => None,
        }
    }

    /// Gravitational parameter GM, km^3/s^2 (DE440 / satellite-solution values).
    pub fn gm(self) -> f64 {
        match self {
            BodyId::Sun => 1.327_124_400_41e11,
            BodyId::Mercury => 2.203_186_9e4,
            BodyId::Venus => 3.248_585_92e5,
            BodyId::Earth => 3.986_004_354_36e5,
            BodyId::Moon => 4.902_800_118e3,
            BodyId::Mars => 4.282_837_37e4,
            BodyId::Jupiter => 1.266_865_34e8,
            BodyId::Saturn => 3.793_118_7e7,
            BodyId::Uranus => 5.793_939e6,
            BodyId::Neptune => 6.835_100e6,
            // Pluto system (barycenter): Pluto + Charon.
            BodyId::Pluto => 9.755e2,
            BodyId::Io => 5.959_9e3,
            BodyId::Europa => 3.202_7e3,
            BodyId::Ganymede => 9.887_8e3,
            BodyId::Callisto => 7.179_3e3,
            BodyId::Titan => 8.978_1e3,
            BodyId::Triton => 1.427_6e3,
            BodyId::Ceres => 62.628,
            BodyId::Vesta => 17.29,
            BodyId::Pallas => 13.63,
            BodyId::Hygiea => 5.8,
        }
    }

    /// Mean radius, km.
    pub fn radius_km(self) -> f64 {
        match self {
            BodyId::Sun => 696_000.0,
            BodyId::Mercury => 2_439.7,
            BodyId::Venus => 6_051.8,
            BodyId::Earth => 6_371.0,
            BodyId::Moon => 1_737.4,
            BodyId::Mars => 3_389.5,
            BodyId::Jupiter => 69_911.0,
            BodyId::Saturn => 58_232.0,
            BodyId::Uranus => 25_362.0,
            BodyId::Neptune => 24_622.0,
            BodyId::Pluto => 1_188.3,
            BodyId::Io => 1_821.6,
            BodyId::Europa => 1_560.8,
            BodyId::Ganymede => 2_634.1,
            BodyId::Callisto => 2_410.3,
            BodyId::Titan => 2_574.7,
            BodyId::Triton => 1_353.4,
            BodyId::Ceres => 469.7,
            BodyId::Vesta => 262.7,
            BodyId::Pallas => 256.0,
            BodyId::Hygiea => 217.0,
        }
    }

    /// Oblateness parameters for the J2 zonal term: `(J2, equatorial radius km,
    /// pole right ascension deg, pole declination deg)` in the ICRF/J2000
    /// frame the ephemeris uses. `None` for bodies whose oblateness is
    /// negligible or whose spin pole isn't modeled (Sun, small moons,
    /// asteroids). Values are standard IAU / planetary-fact-sheet figures.
    pub fn j2_params(self) -> Option<(f64, f64, f64, f64)> {
        Some(match self {
            BodyId::Mercury => (6.0e-5, 2_440.5, 281.0103, 61.4155),
            BodyId::Venus => (4.458e-6, 6_051.8, 272.76, 67.16),
            BodyId::Earth => (1.082_63e-3, 6_378.137, 0.0, 90.0),
            BodyId::Moon => (2.0323e-4, 1_738.1, 269.9949, 66.5392),
            BodyId::Mars => (1.960_45e-3, 3_396.19, 317.681_43, 52.886_50),
            BodyId::Jupiter => (1.4696e-2, 71_492.0, 268.056, 64.495),
            BodyId::Saturn => (1.6291e-2, 60_268.0, 40.589, 83.537),
            BodyId::Uranus => (3.510e-3, 25_559.0, 257.311, -15.175),
            BodyId::Neptune => (3.408e-3, 24_764.0, 299.36, 43.46),
            _ => return None,
        })
    }

    /// Display color (linear-ish RGB).
    pub fn color(self) -> [f32; 3] {
        match self {
            BodyId::Sun => [1.0, 0.85, 0.3],
            BodyId::Mercury => [0.55, 0.5, 0.47],
            BodyId::Venus => [0.9, 0.75, 0.5],
            BodyId::Earth => [0.25, 0.45, 0.9],
            BodyId::Moon => [0.6, 0.6, 0.6],
            BodyId::Mars => [0.85, 0.4, 0.2],
            BodyId::Jupiter => [0.8, 0.65, 0.5],
            BodyId::Saturn => [0.85, 0.75, 0.55],
            BodyId::Uranus => [0.55, 0.8, 0.85],
            BodyId::Neptune => [0.3, 0.4, 0.85],
            BodyId::Pluto => [0.75, 0.68, 0.6],
            BodyId::Io => [0.9, 0.8, 0.4],
            BodyId::Europa => [0.75, 0.7, 0.6],
            BodyId::Ganymede => [0.6, 0.55, 0.5],
            BodyId::Callisto => [0.45, 0.42, 0.4],
            BodyId::Titan => [0.85, 0.65, 0.35],
            BodyId::Triton => [0.7, 0.72, 0.75],
            BodyId::Ceres => [0.6, 0.58, 0.55],
            BodyId::Vesta => [0.65, 0.6, 0.52],
            BodyId::Pallas => [0.55, 0.55, 0.58],
            BodyId::Hygiea => [0.45, 0.45, 0.48],
        }
    }

    /// Orbital period in days (about the parent for moons; heliocentric
    /// otherwise). Negative = retrograde. Used for trail sampling and the
    /// Keplerian fallback for moons.
    pub fn period_days(self) -> f64 {
        match self {
            BodyId::Sun => 1.0,
            BodyId::Mercury => 87.97,
            BodyId::Venus => 224.7,
            BodyId::Earth => 365.25,
            BodyId::Moon => 27.321_661,
            BodyId::Mars => 687.0,
            BodyId::Jupiter => 4_332.6,
            BodyId::Saturn => 10_759.2,
            BodyId::Uranus => 30_688.5,
            BodyId::Neptune => 60_182.0,
            BodyId::Pluto => 90_560.0,
            BodyId::Io => 1.769_138,
            BodyId::Europa => 3.551_181,
            BodyId::Ganymede => 7.154_553,
            BodyId::Callisto => 16.689_017,
            BodyId::Titan => 15.945_42,
            BodyId::Triton => -5.876_854,
            BodyId::Ceres => 1_683.1,
            BodyId::Vesta => 1_325.75,
            BodyId::Pallas => 1_686.0,
            BodyId::Hygiea => 2_030.0,
        }
    }

    /// Mean orbital radius about the parent, km (moons only; fallback model).
    pub fn moon_orbit_radius_km(self) -> f64 {
        match self {
            BodyId::Moon => 384_400.0,
            BodyId::Io => 421_700.0,
            BodyId::Europa => 671_034.0,
            BodyId::Ganymede => 1_070_412.0,
            BodyId::Callisto => 1_882_709.0,
            BodyId::Titan => 1_221_870.0,
            BodyId::Triton => 354_759.0,
            _ => 0.0,
        }
    }

    /// NAIF integer ID for SPICE queries.
    pub fn naif_id(self) -> i32 {
        match self {
            BodyId::Sun => 10,
            BodyId::Mercury => 199,
            BodyId::Venus => 299,
            BodyId::Earth => 399,
            BodyId::Moon => 301,
            BodyId::Mars => 4,
            BodyId::Jupiter => 5,
            BodyId::Saturn => 6,
            BodyId::Uranus => 7,
            BodyId::Neptune => 8,
            BodyId::Pluto => 9,
            BodyId::Io => 501,
            BodyId::Europa => 502,
            BodyId::Ganymede => 503,
            BodyId::Callisto => 504,
            BodyId::Titan => 606,
            BodyId::Triton => 801,
            BodyId::Ceres => 2_000_001,
            BodyId::Vesta => 2_000_004,
            BodyId::Pallas => 2_000_002,
            BodyId::Hygiea => 2_000_010,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `index()` relies on `BodyId`'s declaration order matching `ALL_BODIES`.
    #[test]
    fn index_matches_all_bodies_order() {
        for (i, b) in ALL_BODIES.iter().enumerate() {
            assert_eq!(b.index(), i, "{} out of order", b.name());
        }
    }
}
