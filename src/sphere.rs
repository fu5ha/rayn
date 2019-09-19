use crate::hitable::{HitRecord, Hitable};
use crate::material::Material;
use crate::math::Vec3;
use crate::ray::Ray;

pub struct Sphere<'a> {
    origin: Vec3,
    radius: f32,
    material: &'a dyn Material,
}

impl<'a> Sphere<'a> {
    pub fn new(origin: Vec3, radius: f32, material: &'a dyn Material) -> Self {
        Sphere {
            origin,
            radius,
            material,
        }
    }
    pub fn origin(&self) -> &Vec3 {
        &self.origin
    }
}

impl<'a> Hitable for Sphere<'a> {
    fn hit(&self, ray: &Ray, t_range: ::std::ops::Range<f32>) -> Option<HitRecord> {
        let oc = ray.origin() - self.origin;
        let a = ray.dir().dot(ray.dir().clone());
        let b = 2.0 * oc.dot(ray.dir().clone());
        let c = oc.dot(oc) - self.radius * self.radius;
        let descrim = b * b - 4.0 * a * c;

        if descrim >= 0.0 {
            let desc_sqrt = descrim.sqrt();
            let t = (-b - desc_sqrt) / (2.0 * a);
            if t > t_range.start && t < t_range.end {
                let point = ray.point_at(t);
                let mut offset = point - self.origin();
                offset /= self.radius;
                return Some(HitRecord::new(t, point, offset, self.material));
            }
            let t = (-b + desc_sqrt) / (2.0 * a);
            if t > t_range.start && t < t_range.end {
                let point = ray.point_at(t);
                let mut offset = point - self.origin();
                offset /= self.radius;
                return Some(HitRecord::new(t, point, offset, self.material));
            }
        }
        None
    }
}
