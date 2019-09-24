use crate::material::{ MaterialHandle };
use crate::math::Vec3;
use crate::ray::Ray;

#[derive(Clone, Copy, Debug)]
pub struct Intersection {
    pub t: f32,
    pub point: Vec3,
    pub normal: Vec3,
    pub material: MaterialHandle,
    // pub eta: f32,
}

impl Intersection {
    pub fn new(t: f32, point: Vec3, normal: Vec3, material: MaterialHandle) -> Self {
        Intersection { t, point, normal, material }
    }
}

pub trait Hitable: Send + Sync {
    fn hit(&self, ray: &Ray, time: f32, t_range: ::std::ops::Range<f32>) -> Option<Intersection>;
}

pub struct HitableStore(Vec<Box<dyn Hitable>>);

impl HitableStore {
    pub fn new() -> Self {
        HitableStore(Vec::new())
    }

    pub fn push(&mut self, hitable: Box<dyn Hitable>) {
        self.0.push(hitable)
    }
}

impl ::std::ops::Deref for HitableStore {
    type Target = Vec<Box<dyn Hitable>>;

    fn deref(&self) -> &Vec<Box<dyn Hitable>> {
        &self.0
    }
}

impl Hitable for HitableStore {
    fn hit(&self, ray: &Ray, time: f32, t_range: ::std::ops::Range<f32>) -> Option<Intersection> {
        self.iter()
            .fold((None, t_range.end), |acc, hitable| {
                let mut closest = acc.1;
                let hr = hitable.hit(ray, time, t_range.start..closest);
                if let Some(Intersection { t, .. }) = hr {
                    closest = t;
                }
                let hr = if hr.is_some() { hr } else { acc.0 };
                (hr, closest)
            })
            .0
    }
}
