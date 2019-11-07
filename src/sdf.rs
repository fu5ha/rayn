use crate::animation::WSequenced;
use crate::hitable::{Hitable, WIntersection};
use crate::material::MaterialHandle;
use crate::math::{Transform, Vec3, Wec3};
use crate::ray::{Ray, WRay};
use crate::sphere::Sphere;

use sdfu::*;
use wide::f32x4;

const MAX_MARCHES: u32 = 200;

pub struct TracedSDF<S> {
    sdf: S,
    material: MaterialHandle,
}

impl<S> TracedSDF<S> {
    pub fn new(sdf: S, material: MaterialHandle) -> Self {
        TracedSDF { sdf, material }
    }
}

impl<S: SDF<f32x4, Wec3> + Send + Sync> Hitable for TracedSDF<S> {
    fn hit(&self, ray: &WRay, t_range: ::std::ops::Range<f32x4>) -> f32x4 {
        let dist = self.sdf.dist(ray.origin).abs();
        let mut t = dist;
        for _march in 0..MAX_MARCHES {
            if t.cmp_gt(t_range.end).move_mask() == 0b1111 {
                break;
            }
            let point = ray.point_at(t);
            let dist = self.sdf.dist(point).abs();
            let hit_mask = dist.cmp_lt(t_range.start);
            t = f32x4::merge(hit_mask, t + dist, t);
            if hit_mask.move_mask() == 0b1111 {
                break;
            }
        }
        t
    }

    fn intersection_at(&self, ray: WRay, t: f32x4) -> (MaterialHandle, WIntersection) {
        let normals = self.sdf.normals(f32x4::from(0.005));
        let point = ray.point_at(t);
        let normal = normals.normal_at(point);
        (
            self.material,
            WIntersection::new(ray, t, point, f32x4::from(0.002), normal),
        )
    }
}
