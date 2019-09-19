use crate::math::{ Vec2, Vec3, Transform };
use crate::ray::Ray;
use crate::animation::Sequenced;

pub trait Camera {
    fn get_ray(&self, uv: Vec2, time: f32) -> Ray;
}

pub struct PinholeCamera {
    lower_left: Vec3,
    full_size: Vec3,
    transform_sequence: Box<dyn Sequenced<Transform>>,
}

impl PinholeCamera {
    pub fn new(aspect_ratio: f32, transform_sequence: Box<dyn Sequenced<Transform>>) -> Self {
        PinholeCamera {
            lower_left: Vec3::new(-aspect_ratio * 0.5, -0.5, -1.0),
            full_size: Vec3::new(aspect_ratio, 1.0, 0.0),
            transform_sequence,
        }
    }
}

impl Camera for PinholeCamera {
    fn get_ray(&self, uv: Vec2, time: f32) -> Ray {
        let transform = self.transform_sequence.sample_at(time);
        Ray::new(transform.position, transform.orientation * (self.lower_left + self.full_size * uv).normalized(), 1.0)
    }
}
