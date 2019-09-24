//! This module contains various functions for basic signed distance field primitives.
//! All of them are centered around the origin, so it is necessary to transform the point into
//! object local space of the primitive to use them.
// use crate::animation::Sequenced;
use crate::hitable::{ Hitable, Intersection };
use crate::math::{ Vec3 };
use crate::material::{ MaterialHandle };
use crate::ray::Ray;
use sdfu::*;

const HIT_THRESHOLD: f32 = 0.0001;

pub struct TracedSDF<S> {
    sdf: S,
    material: MaterialHandle,
}

impl<S> TracedSDF<S> {
    pub fn new(sdf: S, material: MaterialHandle) -> Self {
        TracedSDF { sdf, material }
    }
}

impl<S: SDF<f32, Vec3> + Send + Sync> Hitable for TracedSDF<S> {
    fn hit(&self, ray: &Ray, _time: f32, t_range: ::std::ops::Range<f32>) -> Option<Intersection> {
        let dist = self.sdf.dist(*ray.origin());
        if dist < t_range.end {
            let mut t = dist;
            let mut last_dist = dist;
            loop {
                let point = ray.point_at(t);
                let dist = self.sdf.dist(point);
                if dist.abs() < HIT_THRESHOLD {
                    // t += dist;
                    let point = ray.point_at(t);
                    let normals = self.sdf.normals(0.005);
                    let normal = normals.normal_at(point) * last_dist.signum();
                    let point = point + normal * 2.0 * HIT_THRESHOLD;
                    return Some(Intersection::new(t, point, normal, self.material));
                }
                t += dist;
                last_dist = dist;
                if t > t_range.end {
                    break;
                }
            }
        }
        None
    }
}