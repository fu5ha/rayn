use crate::math::{f32x4, Vec2u, Vec3, Wec3};
use crate::spectrum::{Srgb, WSrgb};

macro_rules! rays {
    ($($n:ident => $t:ident, $st:ident, $tt:ident, $tc:ty),+) => {
        $(#[derive(Clone, Copy, Debug)]
        pub struct $n {
            pub time: $tt,
            pub origin: $t,
            pub dir: $t,
            pub radiance: $st,
            pub throughput: $st,
            pub tile_coord: $tc,
        }

        impl $n {
            pub fn new(origin: $t, dir: $t, time: $tt, tile_coord: $tc) -> Self {
                Self { time, origin, dir, radiance: $st::zero(), throughput: $st::one(), tile_coord, }
            }

            #[allow(dead_code)]
            pub fn point_at(&self, t: $tt) -> $t {
                self.dir.mul_add($t::new(t, t, t), self.origin)
            }
        })+
    }
}

rays!(Ray => Vec3, Srgb, f32, Vec2u, WRay => Wec3, WSrgb, f32x4, [Vec2u; 4]);

impl From<[Ray; 4]> for WRay {
    fn from(rays: [Ray; 4]) -> Self {
        Self {
            time: f32x4::new(rays[0].time, rays[1].time, rays[2].time, rays[3].time),
            origin: Wec3::from([
                rays[0].origin,
                rays[1].origin,
                rays[2].origin,
                rays[3].origin,
            ]),
            dir: Wec3::from([rays[0].dir, rays[1].dir, rays[2].dir, rays[3].dir]),
            radiance: WSrgb::from([
                rays[0].radiance,
                rays[1].radiance,
                rays[2].radiance,
                rays[3].radiance,
            ]),
            throughput: WSrgb::from([
                rays[0].throughput,
                rays[1].throughput,
                rays[2].throughput,
                rays[3].throughput,
            ]),
            tile_coord: [
                rays[0].tile_coord,
                rays[1].tile_coord,
                rays[2].tile_coord,
                rays[3].tile_coord,
            ],
        }
    }
}

impl Into<[Ray; 4]> for WRay {
    fn into(self) -> [Ray; 4] {
        let times = self.time.as_ref();
        let origins: [Vec3; 4] = self.origin.into();
        let dirs: [Vec3; 4] = self.dir.into();
        let throughputs: [Srgb; 4] = self.throughput.into();
        let radiances: [Srgb; 4] = self.radiance.into();
        [
            Ray {
                time: times[0],
                origin: origins[0],
                dir: dirs[0],
                radiance: radiances[0],
                throughput: throughputs[0],
                tile_coord: self.tile_coord[0],
            },
            Ray {
                time: times[0],
                origin: origins[1],
                dir: dirs[1],
                radiance: radiances[1],
                throughput: throughputs[1],
                tile_coord: self.tile_coord[1],
            },
            Ray {
                time: times[0],
                origin: origins[2],
                dir: dirs[2],
                radiance: radiances[2],
                throughput: throughputs[2],
                tile_coord: self.tile_coord[2],
            },
            Ray {
                time: times[0],
                origin: origins[3],
                dir: dirs[3],
                radiance: radiances[3],
                throughput: throughputs[3],
                tile_coord: self.tile_coord[3],
            },
        ]
    }
}
