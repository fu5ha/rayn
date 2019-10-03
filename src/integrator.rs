use dynamic_arena::{DynamicArena, NonSend};
use rand::rngs::SmallRng;
use rand::Rng;

use crate::hitable::{Hitable, Intersection};
use crate::math::Vec3;
use crate::ray::Ray;
use crate::spectrum::{IsSpectrum, Rgb};
use crate::world::World;

pub trait Integrator: Send + Sync {
    fn integrate<S: IsSpectrum>(
        &self,
        world: &World<S>,
        ray: Ray,
        time: f32,
        rng: &mut SmallRng,
        arena: &DynamicArena<'_, NonSend>,
    ) -> (S, Option<Intersection>);
}

#[derive(Clone, Copy)]
pub struct PathTracingIntegrator {
    pub max_bounces: usize,
}

impl Integrator for PathTracingIntegrator {
    fn integrate<S: IsSpectrum>(
        &self,
        world: &World<S>,
        mut ray: Ray,
        time: f32,
        rng: &mut SmallRng,
        arena: &DynamicArena<'_, NonSend>,
    ) -> (S, Option<Intersection>) {
        let mut radiance = S::zero();
        let mut throughput = S::one();
        let mut first_intersection = None;
        for bounce in 0.. {
            if let Some(mut intersection) = world.hitables.hit(&ray, time, 0.001..1000.0) {
                let wi = *ray.dir();
                let material = world.materials.get(intersection.material);

                let bsdf = material.setup_scattering_functions(&mut intersection, &arena);

                radiance += bsdf.le(-wi, &mut intersection) * throughput;
                let scattering_event = bsdf.scatter(wi, &mut intersection, rng);

                if bounce == 0 {
                    first_intersection = Some(intersection);
                }

                if let Some(se) = scattering_event {
                    let ndl = se.wi.dot(intersection.normal).abs();
                    if ndl == 0.0 || se.pdf == 0.0 || se.f.is_black() {
                        break;
                    }
                    throughput *= se.f / se.pdf * ndl;
                    if throughput.is_nan() {
                        break;
                    }
                    ray = intersection.create_ray(se.wi);
                } else {
                    break;
                }
            } else {
                let dir = ray.dir();
                let t = 0.5 * (dir.y + 1.0);

                let l = throughput
                    * S::from(Rgb::from(Vec3::lerp(
                        Vec3::one(),
                        Vec3::new(0.5, 0.7, 1.0),
                        t,
                    )));
                radiance += l;
                break;
            }

            if bounce >= self.max_bounces {
                break;
            }

            if bounce > 2 {
                let roulette_factor = (1.0 - throughput.max_channel()).max(0.05);
                if rng.gen::<f32>() < roulette_factor {
                    break;
                }
                throughput /= 1.0 - roulette_factor;
            }
        }
        (radiance, first_intersection)
    }
}
