use crate::animation::Sequenced;
use crate::hitable::{ Intersection, Hitable };
use crate::material::{ MaterialHandle };
use crate::math::{ Transform };
use crate::ray::Ray;

pub struct Sphere<TR: Sequenced<Transform>> {
    transform_seq: TR,
    radius: f32,
    material: MaterialHandle,
}

impl<TR: Sequenced<Transform>> Sphere<TR> {
    pub fn new(transform_seq: TR, radius: f32, material: MaterialHandle) -> Self {
        Sphere {
            transform_seq,
            radius,
            material,
        }
    }
}

impl<'a, TR: Sequenced<Transform>> Hitable for Sphere<TR> {
    fn hit(&self, ray: &Ray, time: f32, t_range: ::std::ops::Range<f32>) -> Option<Intersection> {
        let transform = self.transform_seq.sample_at(time);
        let origin = transform.position;
        let oc = ray.origin() - origin;
        let a = ray.dir().magnitude_squared();
        let b = 2.0 * oc.dot(ray.dir().clone());
        let c = oc.magnitude_squared() - self.radius * self.radius;
        let descrim = b * b - 4.0 * a * c;

        if descrim >= 0.0 {
            let desc_sqrt = descrim.sqrt();
            let t = (-b - desc_sqrt) / (2.0 * a);
            if t > t_range.start && t < t_range.end {
                let point = ray.point_at(t);
                let mut offset = point - origin;
                offset /= self.radius;
                return Some(Intersection::new(t, point, offset, self.material));
            }
            let t = (-b + desc_sqrt) / (2.0 * a);
            if t > t_range.start && t < t_range.end {
                let point = ray.point_at(t);
                let mut offset = point - origin;
                offset /= self.radius;
                return Some(Intersection::new(t, point, offset, self.material));
            }
        }
        None
    }
}
