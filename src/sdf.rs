// use crate::animation::Sequenced;
use crate::hitable::{ Hitable, Intersection };
use crate::math::{ Vec3 };
use crate::material::{ MaterialHandle };
use crate::ray::Ray;
use crate::spectrum::IsSpectrum;

use sdfu::*;

const MAX_MARCHES: u32 = 1000;

pub struct TracedSDF<S> {
    sdf: S,
    material: MaterialHandle,
}

impl<S> TracedSDF<S> {
    pub fn new(sdf: S, material: MaterialHandle) -> Self {
        TracedSDF { sdf, material }
    }
}

impl<SD: SDF<f32, Vec3> + Send + Sync, SP: IsSpectrum> Hitable<SP> for TracedSDF<SD> {
    fn hit(&self, ray: &Ray, _time: f32, t_range: ::std::ops::Range<f32>) -> Option<Intersection<SP>> {
        let dist = self.sdf.dist(*ray.origin()).abs();
        if dist < t_range.end {
            let mut t = dist;
            for _march in 0..MAX_MARCHES {
                let point = ray.point_at(t);
                let dist = self.sdf.dist(point).abs();
                if dist < t_range.start {
                    let point = ray.point_at(t);
                    let normals = self.sdf.normals(0.005);
                    let normal = normals.normal_at(point);
                    return Some(Intersection::new(t, point, t_range.start * 2.0, normal, self.material));
                }

                t += dist;
                if t > t_range.end {
                    break;
                }
            }
        }
        None
    }
}