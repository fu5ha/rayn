use crate::hitable::{Hitable, WHit, WShadingPoint};
use crate::material::MaterialHandle;
use crate::math::Wec3;
use crate::ray::WRay;

use sdfu::*;
use wide::f32x4;

const MAX_MARCHES: u32 = 100;

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

    fn get_shading_info(&self, hit: WHit) -> (MaterialHandle, WShadingPoint) {
        let normals = self.sdf.normals_fast(f32x4::from(0.001));
        let point = hit.point();
        let normal = normals.normal_at(point);
        (
            self.material,
            WShadingPoint::new(hit, point, f32x4::from(0.002), normal),
        )
    }
}
