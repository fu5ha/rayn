use crate::animation::Sequenced;
use crate::hitable::{Hitable, Intersection};
use crate::material::MaterialHandle;
use crate::math::{Transform, Vec3};
use crate::ray::Ray;
use crate::sphere::Sphere;

use sdfu::*;

const MAX_MARCHES: u32 = 1000;

pub struct TracedSDF<S, TR: Sequenced<Transform>> {
    sdf: S,
    bounding_sphere: Sphere<TR>,
    material: MaterialHandle,
}

impl<S, TR: Sequenced<Transform>> TracedSDF<S, TR> {
    pub fn new(sdf: S, bounding_sphere: Sphere<TR>, material: MaterialHandle) -> Self {
        TracedSDF {
            sdf,
            bounding_sphere,
            material,
        }
    }
}

impl<S: Sdf<f32, Vec3> + Send + Sync, TR: Sequenced<Transform>> Hitable for TracedSDF<S, TR> {
    fn hit(&self, ray: &Ray, time: f32, t_range: ::std::ops::Range<f32>) -> Option<Intersection> {
        if let Some(_) = self.bounding_sphere.hit(ray, time, t_range.clone()) {
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
                        return Some(Intersection::new(
                            t,
                            point,
                            t_range.start * 2.0,
                            normal,
                            self.material,
                        ));
                    }

                    t += dist;
                    if t > t_range.end {
                        break;
                    }
                }
            }
        }
        None
    }
}
