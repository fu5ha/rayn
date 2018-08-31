use super::Vec3;

pub struct Ray {
    orig: Vec3,
    dir: Vec3,
}

impl Ray {
    pub fn new(orig: Vec3, dir: Vec3) -> Self { Ray { orig, dir } }
    pub fn orig(&self) -> &Vec3 { &self.orig }
    pub fn dir(&self) -> &Vec3 { &self.dir }
    pub fn point_at(&self, t: f32) -> Vec3 {
        self.dir.mul_add(Vec3::new(t,t,t), self.orig)
    }
}
