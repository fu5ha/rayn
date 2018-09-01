use std::sync::Arc;

use material::Material;
use math::Vec3;
use ray::Ray;

#[derive(Clone)]
pub struct HitRecord {
    pub t: f32,
    pub p: Vec3,
    pub n: Vec3,
    pub material: Arc<Material>,
}

impl HitRecord {
    pub fn new(t: f32, p: Vec3, n: Vec3, material: Arc<Material>) -> Self {
        HitRecord { t, p, n, material }
    }
}

pub trait Hitable: Send + Sync {
    fn hit(&self, ray: &Ray, t_range: ::std::ops::Range<f32>) -> Option<HitRecord>;
}

pub struct HitableList(Vec<Box<Hitable>>);

impl HitableList {
    pub fn new() -> Self {
        HitableList(Vec::new())
    }

    pub fn push(&mut self, hitable: Box<Hitable>) {
        self.0.push(hitable)
    }
}

impl ::std::ops::Deref for HitableList {
    type Target = Vec<Box<Hitable>>;

    fn deref(&self) -> &Vec<Box<Hitable>> {
        &self.0
    }
}

impl Hitable for HitableList {
    fn hit(&self, ray: &Ray, t_range: ::std::ops::Range<f32>) -> Option<HitRecord> {
        let ret = self
            .iter()
            .fold((None, t_range.end), |acc, hitable| {
                let mut closest = acc.1;
                let hr = hitable.hit(ray, t_range.start..closest);
                if let Some(HitRecord {
                    t,
                    p: _,
                    n: _,
                    material: _,
                }) = hr
                {
                    closest = t;
                }
                let hr = if hr.is_some() { hr } else { acc.0 };
                (hr, closest)
            }).0;
        ret
    }
}
