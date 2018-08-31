use ray::Ray;
use super::Vec3;

pub struct Camera {
    lower_left: Vec3,
    full_size: Vec3,
    origin: Vec3,
}

impl Camera {
    pub fn new(aspect_ratio: f32) -> Self {
        Camera {
            lower_left: Vec3::new(-aspect_ratio, -1.0, -1.0),
            full_size: Vec3::new(aspect_ratio * 2.0, 2.0, 0.0),
            origin: Vec3::zero(),
        }
    }

    pub fn get_ray(&self, uv: Vec3) -> Ray {
        Ray::new(self.origin.clone(), self.lower_left + self.full_size * uv)
    }
}