use crate::animation::Sequenced;
use crate::hitable::{HitRecord, Hitable};
use crate::material::Material;
use crate::math::{ Transform };
use crate::ray::Ray;

pub struct Sphere<'a> {
    transform_seq: Box<dyn Sequenced<Transform>>,
    radius: f32,
    material: &'a dyn Material,
}

impl<'a> Sphere<'a> {
    pub fn new(transform_seq: Box<dyn Sequenced<Transform>>, radius: f32, material: &'a dyn Material) -> Self {
        Sphere {
            transform_seq,
            radius,
            material,
        }
    }
}

impl<'a> Hitable for Sphere<'a> {
    fn hit(&self, ray: &Ray, time: f32, t_range: ::std::ops::Range<f32>) -> Option<HitRecord> {
        let transform = self.transform_seq.sample_at(time);
        let origin = transform.position;
        let oc = ray.origin() - origin;
        let a = ray.dir().dot(ray.dir().clone());
        let b = 2.0 * oc.dot(ray.dir().clone());
        let c = oc.dot(oc) - self.radius * self.radius;
        let descrim = b * b - 4.0 * a * c;

        if descrim >= 0.0 {
            let desc_sqrt = descrim.sqrt();
            let t = (-b - desc_sqrt) / (2.0 * a);
            if t > t_range.start && t < t_range.end {
                let point = ray.point_at(t);
                let mut offset = point - origin;
                offset /= self.radius;
                return Some(HitRecord::new(t, point, offset, self.material));
            }
            let t = (-b + desc_sqrt) / (2.0 * a);
            if t > t_range.start && t < t_range.end {
                let point = ray.point_at(t);
                let mut offset = point - origin;
                offset /= self.radius;
                return Some(HitRecord::new(t, point, offset, self.material));
            }
        }
        None
    }
}
