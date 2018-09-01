use math::Vec3;
use ray::Ray;

pub struct Camera {
    lower_left: Vec3,
    full_size: Vec3,
    origin: Vec3,
}

impl Camera {
    pub fn new(aspect_ratio: f32) -> Self {
        Camera {
            lower_left: Vec3::new(-aspect_ratio * 0.5, -0.5, -1.0),
            full_size: Vec3::new(aspect_ratio, 1.0, 0.0),
            origin: Vec3::new(0.0, 0.0, 1.0),
        }
    }

    pub fn get_ray(&self, uv: Vec3) -> Ray {
        Ray::new(self.origin.clone(), self.lower_left + self.full_size * uv)
    }
}