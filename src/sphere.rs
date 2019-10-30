use crate::animation::{Sequenced, WSequenced};
use crate::hitable::{Hitable, Intersection};
use crate::material::MaterialHandle;
use crate::math::{Vec3, Wec3};
use crate::ray::{Ray, WRay};

use wide::f32x4;

pub struct Sphere<TR> {
    transform_seq: TR,
    radius: f32,
    material: MaterialHandle,
}

impl<TR> Sphere<TR> {
    pub fn new(transform_seq: TR, radius: f32, material: MaterialHandle) -> Self {
        Sphere {
            transform_seq,
            radius,
            material,
        }
    }
}

impl<TR: WSequenced<Wec3> + Sequenced<Vec3>> Hitable for Sphere<TR> {
    fn hit(&self, ray: &WRay, t_range: ::std::ops::Range<f32x4>) -> f32x4 {
        let origin = WSequenced::sample_at(&self.transform_seq, ray.time);
        let oc = ray.origin - origin;
        let a = ray.dir.mag_sq();
        let b = f32x4::from(2.0) * oc.dot(ray.dir);
        let c = oc.mag_sq() - f32x4::from(self.radius * self.radius);
        let descrim = b * b - f32x4::from(4.0) * a * c;

        let desc_sqrt = descrim.sqrt();

        let t1 = (-b - desc_sqrt) / (f32x4::from(2.0) * a);

        let t2 = (-b + desc_sqrt) / (f32x4::from(2.0) * a);

        let mask = t1.cmp_le(t2);

        let t = f32x4::merge(mask, t2, t1);
        let min_mask = t.cmp_gt(t_range.start);
        let max_mask = t.cmp_le(t_range.end);

        let miss = f32x4::from(wide::consts::MAX);

        f32x4::merge(max_mask, miss, f32x4::merge(min_mask, miss, t))
    }

    fn intersection_at(&self, ray: Ray, t: f32, point: Vec3) -> (MaterialHandle, Intersection) {
        let origin = Sequenced::sample_at(&self.transform_seq, ray.time);
        let normal = (point - origin).normalized();
        (self.material, Intersection::new(ray, t, point, 0.0, normal))
    }
}
