use crate::math::Vec3;

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    origin: Vec3,
    dir: Vec3,
    medium_ior: f32,
}

impl Ray {
    pub fn new(origin: Vec3, dir: Vec3, medium_ior: f32) -> Self {
        Ray { origin, dir, medium_ior }
    }

    pub fn origin(&self) -> &Vec3 {
        &self.origin
    }

    pub fn dir(&self) -> &Vec3 {
        &self.dir
    }

    pub fn medium_ior(&self) -> f32 {
        self.medium_ior
    }

    pub fn point_at(&self, t: f32) -> Vec3 {
        self.dir.mul_add(Vec3::new(t, t, t), self.origin)
    }
}
