use rand::rngs::SmallRng;

use crate::animation::WSequenced;
use crate::math::{RandomSample2d, Transform, Vec2, Vec2u, Vec3, Wec2, Wec3};
use crate::ray::WRay;

use wide::f32x4;

pub trait Camera: Send + Sync {
    fn get_rays(&self, tile_coord: Vec2u, uv: Wec2, time: f32x4, rng: &mut SmallRng) -> WRay;
}

#[derive(Clone, Copy, Debug)]
pub struct CameraHandle(usize);

pub struct CameraStore(Vec<Box<dyn Camera>>);

impl CameraStore {
    pub fn new() -> Self {
        CameraStore(Vec::new())
    }

    pub fn add_camera(&mut self, material: Box<dyn Camera>) -> CameraHandle {
        self.0.push(material);
        CameraHandle(self.0.len() - 1)
    }

    pub fn get(&self, handle: CameraHandle) -> &dyn Camera {
        self.0.get(handle.0).map(|b| b.as_ref()).unwrap()
    }
}

#[derive(Copy, Clone)]
pub struct PinholeCamera<OS> {
    lower_left: Wec3,
    full_size: Wec3,
    origin_sequence: OS,
}

impl<TR> PinholeCamera<TR> {
    #[allow(dead_code)]
    pub fn new(aspect_ratio: f32, origin_sequence: TR) -> Self {
        PinholeCamera {
            lower_left: Wec3::splat(Vec3::new(-aspect_ratio * 0.5, -0.5, -1.0)),
            full_size: Wec3::splat(Vec3::new(aspect_ratio, 1.0, 0.0)),
            origin_sequence,
        }
    }
}

impl<OS: WSequenced<Wec3>> Camera for PinholeCamera<OS> {
    fn get_rays(&self, tile_coord: Vec2u, uv: Wec2, time: f32x4, _rng: &mut SmallRng) -> WRay {
        let origin = self.origin_sequence.sample_at(time);
        WRay::new(
            origin,
            (self.lower_left + self.full_size * Wec3::from(uv)).normalized(),
            time,
            [tile_coord, tile_coord, tile_coord, tile_coord],
        )
    }
}

#[derive(Clone, Copy)]
pub struct ThinLensCamera<A, O, LA, U, F> {
    half_size: Wec2,
    aperture: A,
    origin: O,
    at: LA,
    up: U,
    focus: F,
}

impl<A, O, LA, U, F> ThinLensCamera<A, O, LA, U, F> {
    pub fn new(aspect: f32, vfov: f32, aperture: A, origin: O, at: LA, up: U, focus: F) -> Self {
        let theta = vfov * std::f32::consts::PI / 180.0;
        let half_height = (theta / 2.0).tan();
        let half_width = aspect * half_height;
        ThinLensCamera {
            half_size: Wec2::splat(Vec2::new(half_width, half_height)),
            aperture,
            origin,
            at,
            up,
            focus,
        }
    }
}

impl<A, O, LA, U, F> Camera for ThinLensCamera<A, O, LA, U, F>
where
    A: WSequenced<f32x4>,
    O: WSequenced<Wec3>,
    LA: WSequenced<Wec3>,
    U: WSequenced<Wec3>,
    F: WSequenced<Wec3>,
{
    fn get_rays(&self, tile_coord: Vec2u, uv: Wec2, time: f32x4, rng: &mut SmallRng) -> WRay {
        let origin = self.origin.sample_at(time);
        let at = self.at.sample_at(time);
        let up = self.up.sample_at(time);
        let focus = self.focus.sample_at(time);
        let focus_dist = (focus - origin).mag();
        let aperture = self.aperture.sample_at(time);

        let basis_w = (origin - at).normalized();
        let basis_u = up.cross(basis_w).normalized();
        let basis_v = basis_w.cross(basis_u);
        let lower_left = origin
            - basis_u * self.half_size.x * focus_dist
            - basis_v * self.half_size.y * focus_dist
            - basis_w * focus_dist;

        let horiz = basis_u * self.half_size.x * focus_dist * f32x4::from(2.0) * uv.x;
        let verti = basis_v * self.half_size.y * focus_dist * f32x4::from(2.0) * uv.y;

        let rd = Wec2::rand_in_unit_disk(rng) * aperture;
        let offset = basis_u * rd.x + basis_v * rd.y;

        let origin = origin + offset;
        WRay::new(
            origin,
            (lower_left + horiz + verti - origin).normalized(),
            time,
            [tile_coord, tile_coord, tile_coord, tile_coord],
        )
    }
}
