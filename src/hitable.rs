use crate::material::Material;
use crate::math::Vec3;
use crate::ray::Ray;

#[derive(Clone)]
pub struct HitRecord<'a> {
    pub t: f32,
    pub point: Vec3,
    pub normal: Vec3,
    pub material: &'a dyn Material,
}

impl<'a> HitRecord<'a> {
    pub fn new(t: f32, point: Vec3, normal: Vec3, material: &'a dyn Material) -> Self {
        HitRecord { t, point, normal, material }
    }
}

pub trait Hitable: Send + Sync {
    fn hit(&self, ray: &Ray, t_range: ::std::ops::Range<f32>) -> Option<HitRecord>;
}

pub struct HitableList(Vec<Box<dyn Hitable>>);

impl HitableList {
    pub fn new() -> Self {
        HitableList(Vec::new())
    }

    pub fn push(&mut self, hitable: Box<dyn Hitable>) {
        self.0.push(hitable)
    }
}

impl ::std::ops::Deref for HitableList {
    type Target = Vec<Box<dyn Hitable>>;

    fn deref(&self) -> &Vec<Box<dyn Hitable>> {
        &self.0
    }
}

impl Hitable for HitableList {
    fn hit(&self, ray: &Ray, t_range: ::std::ops::Range<f32>) -> Option<HitRecord> {
        self.iter()
            .fold((None, t_range.end), |acc, hitable| {
                let mut closest = acc.1;
                let hr = hitable.hit(ray, t_range.start..closest);
                if let Some(HitRecord { t, .. }) = hr {
                    closest = t;
                }
                let hr = if hr.is_some() { hr } else { acc.0 };
                (hr, closest)
            })
            .0
    }
}
