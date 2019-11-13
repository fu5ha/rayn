use crate::hitable::{Hitable, WHit, WShadingPoint};
use crate::material::MaterialHandle;
use crate::math::{f32x4, Wec3};
use crate::ray::WRay;

use sdfu::*;

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
        let normals = self.sdf.normals_fast(f32x4::from(0.0005));
        let point = hit.point();
        let normal = normals.normal_at(point);
        (
            self.material,
            WShadingPoint::new(hit, point, f32x4::from(0.002), normal),
        )
    }
}

#[derive(Clone, Copy)]
pub struct MandelBox {
    iterations: usize,
    scale: f32x4,
    scale_vec: Wec3,
    box_fold: BoxFold,
    sphere_fold: SphereFold,
}

impl MandelBox {
    pub fn new(iterations: usize, box_fold: BoxFold, sphere_fold: SphereFold, scale: f32) -> Self {
        Self {
            iterations,
            box_fold,
            sphere_fold,
            scale: scale.into(),
            scale_vec: Wec3::broadcast(scale.into()),
        }
    }
}

impl SDF<f32x4, Wec3> for MandelBox {
    fn dist(&self, mut p: Wec3) -> f32x4 {
        let offset = p;
        let one = f32x4::from(1.0);
        let mut dr = one;
        for _ in 0..self.iterations {
            self.box_fold.box_fold(&mut p);
            self.sphere_fold.sphere_fold(&mut p, &mut dr);

            p = p.mul_add(self.scale_vec, offset);
            dr = dr.mul_add(self.scale.abs(), one);
        }

        let d = p.mag() / dr.abs();
        // println!("{}", d);
        d
    }
}

#[derive(Clone, Copy)]
pub struct BoxFold {
    l: Wec3,
    neg_l: Wec3,
    two: Wec3,
}

impl BoxFold {
    pub fn new(side_length: f32) -> Self {
        let l = Wec3::broadcast(side_length.into());
        BoxFold {
            l,
            neg_l: -l,
            two: Wec3::broadcast(2.0.into()),
        }
    }

    pub fn box_fold(&self, point: &mut Wec3) {
        *point = point.clamped(self.neg_l, self.l).mul_add(self.two, -*point)
    }
}

#[derive(Clone, Copy)]
pub struct SphereFold {
    rad_sq: f32x4,
}

impl SphereFold {
    pub fn new(radius: f32) -> Self {
        Self {
            rad_sq: radius.into(),
        }
    }

    pub fn sphere_fold(&self, point: &mut Wec3, dr: &mut f32x4) {
        let r2 = point.mag_sq();
        let mul = (self.rad_sq / r2)
            .max(self.rad_sq)
            .clamp(f32x4::from(0.0), f32x4::from(1.0));
        *point *= mul;
        *dr *= mul;
    }
}
