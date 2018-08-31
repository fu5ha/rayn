use hitable::{Hitable, HitRecord};
use ray::Ray;
use super::Vec3;

pub struct Sphere {
    orig: Vec3,
    rad: f32,
}

impl Sphere {
    pub fn new(orig: Vec3, rad: f32) -> Self { Sphere { orig, rad } }
    pub fn orig(&self) -> &Vec3 { &self.orig }
}

impl Hitable for Sphere {
    fn hit(&self, ray: &Ray, t_range: ::std::ops::Range<f32>) -> Option<HitRecord> {
        let oc = ray.orig() - self.orig;
        let a = ray.dir().dot(ray.dir().clone());
        let b = 2.0 * oc.clone().dot(ray.dir().clone());
        let c = oc.clone().dot(oc) - self.rad * self.rad;
        let descrim = b*b - 4.0*a*c;

        if descrim >= 0.0 {
            let desc_sqrt = descrim.sqrt();
            let t = (-b - desc_sqrt) / (2.0 * a);
            if t > t_range.start && t < t_range.end {
                let p = ray.point_at(t);
                let mut n = p - self.orig();
                n /= self.rad;
                return Some(HitRecord::new(t, p, n));
            }
            let t = (-b + desc_sqrt) / (2.0 * a);
            if t > t_range.start && t < t_range.end {
                let p = ray.point_at(t);
                let mut n = p - self.orig();
                n /= self.rad;
                return Some(HitRecord::new(t, p, n));
            }
        }
        None
    }
}