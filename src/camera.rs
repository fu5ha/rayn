use rand::prelude::*;

use crate::animation::Sequenced;
use crate::math::{RandomSample2d, Transform, Vec2, Vec3};
use crate::ray::Ray;

pub trait Camera: Send + Sync {
    fn get_ray(&self, uv: Vec2, time: f32, rng: &mut ThreadRng) -> Ray;
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
pub struct PinholeCamera<TR> {
    lower_left: Vec3,
    full_size: Vec3,
    transform_sequence: TR,
}

impl<TR> PinholeCamera<TR> {
    #[allow(dead_code)]
    pub fn new(aspect_ratio: f32, transform_sequence: TR) -> Self {
        PinholeCamera {
            lower_left: Vec3::new(-aspect_ratio * 0.5, -0.5, -1.0),
            full_size: Vec3::new(aspect_ratio, 1.0, 0.0),
            transform_sequence,
        }
    }
}

impl<TR: Sequenced<Transform>> Camera for PinholeCamera<TR> {
    fn get_ray(&self, uv: Vec2, time: f32, _rng: &mut ThreadRng) -> Ray {
        let transform = self.transform_sequence.sample_at(time);
        Ray::new(
            transform.position,
            transform.orientation * (self.lower_left + self.full_size * uv).normalized(),
        )
    }
}

#[derive(Clone, Copy)]
pub struct ThinLensCamera<A, O, LA, U, F> {
    half_size: Vec2,
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
            half_size: Vec2::new(half_width, half_height),
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
    A: Sequenced<f32>,
    O: Sequenced<Vec3>,
    LA: Sequenced<Vec3>,
    U: Sequenced<Vec3>,
    F: Sequenced<Vec3>,
{
    fn get_ray(&self, uv: Vec2, time: f32, rng: &mut ThreadRng) -> Ray {
        let origin = self.origin.sample_at(time);
        let at = self.at.sample_at(time);
        let up = self.up.sample_at(time);
        let focus = self.focus.sample_at(time);
        let focus_dist = (focus - origin).magnitude();
        let aperture = self.aperture.sample_at(time);

        let basis_w = (origin - at).normalized();
        let basis_u = up.cross(basis_w).normalized();
        let basis_v = basis_w.cross(basis_u);
        let lower_left = origin
            - basis_u * self.half_size.x * focus_dist
            - basis_v * self.half_size.y * focus_dist
            - basis_w * focus_dist;

        let horiz = basis_u * self.half_size.x * focus_dist * 2.0 * uv.x;
        let verti = basis_v * self.half_size.y * focus_dist * 2.0 * uv.y;

        let rd = Vec2::rand_in_unit_disk(rng) * aperture;
        let offset = basis_u * rd.x + basis_v * rd.y;

        let origin = origin + offset;
        Ray::new(origin, (lower_left + horiz + verti - origin).normalized())
    }
}
