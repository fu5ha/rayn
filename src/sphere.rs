use crate::animation::{Sequenced, WSequenced};
use crate::hitable::{Hitable, WIntersection};
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

impl<TR: WSequenced<Wec3>> Hitable for Sphere<TR> {
    fn hit(&self, ray: &WRay, t_range: ::std::ops::Range<f32x4>) -> f32x4 {
        let origin = WSequenced::sample_at(&self.transform_seq, ray.time);
        let oc = ray.origin - origin;
        let a = ray.dir.mag_sq();
        let b = f32x4::from(2.0) * oc.dot(ray.dir);
        let c = oc.mag_sq() - f32x4::from(self.radius * self.radius);
        let descrim = b * b - f32x4::from(4.0) * a * c;

        let desc_pos = descrim.cmp_gt(f32x4::from(0.0));

        let miss = f32x4::from(wide::consts::MAX);

        if desc_pos.move_mask() != 0b0000 {
            let desc_sqrt = descrim.sqrt();

            let t1 = (-b - desc_sqrt) / (f32x4::from(2.0) * a);
            let t1_valid = t1.cmp_gt(t_range.start) & t1.cmp_le(t_range.end) & desc_pos;

            let t2 = (-b + desc_sqrt) / (f32x4::from(2.0) * a);
            let t2_valid = t2.cmp_gt(t_range.start) & t2.cmp_le(t_range.end) & desc_pos;

            let take_t1 = t1.cmp_lt(t2) & t1_valid;

            let t = f32x4::merge(take_t1, t2, t1);

            f32x4::merge(t1_valid | t2_valid, miss, t)
        } else {
            miss
        }
    }

    fn intersection_at(&self, ray: WRay, t: f32x4) -> (MaterialHandle, WIntersection) {
        let point = ray.point_at(t);
        let origin = WSequenced::sample_at(&self.transform_seq, ray.time);
        let normal = (point - origin).normalized();
        (
            self.material,
            WIntersection::new(ray, t, point, f32x4::from(0.0), normal),
        )
    }
}
