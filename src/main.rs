use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use rand::distributions::Uniform;
use rand::prelude::*;
use rayon::prelude::*;

mod animation;
mod camera;
mod spectrum;
mod hitable;
mod material;
mod math;
mod ray;
mod sdf;
mod sphere;
mod world;

use animation::{ Sequence, TransformSequence };
use camera::{ Camera, PinholeCamera };
use spectrum::Spectrum;
use hitable::{Hitable, HitableStore};
use material::{Dielectric, MaterialStore, Metal, Refractive};
use math::{ Vec2, Vec3, Quat };
use ray::Ray;
use sdf::TracedSDF;
use sphere::Sphere;
use world::World;

use sdfu::SDF;

const DIMS: (u32, u32) = (1920, 1080);
const SAMPLES: usize = 256;
const MAX_BOUNCES: usize = 8;

fn setup() -> World {
    let mut materials = MaterialStore::new();
    let pink_diffuse = materials.add_material(Box::new(Dielectric::new(Spectrum::new(0.9, 0.35, 0.55), 0.0)));
    let ground = materials.add_material(Box::new(Dielectric::new(Spectrum::new(0.25, 0.2, 0.35), 0.3)));
    let gold = materials.add_material(Box::new(Metal::new(Spectrum::new(1.0, 0.9, 0.5), 0.0)));
    let gold_rough = materials.add_material(Box::new(Metal::new(Spectrum::new(1.0, 0.9, 0.5), 0.2)));
    let silver = materials.add_material(Box::new(Metal::new(Spectrum::new(0.9, 0.9, 0.9), 0.1)));
    let glass = materials.add_material(Box::new(Refractive::new(Spectrum::new(0.9, 0.9, 0.9), 0.0, 1.2)));
    let glass_rough = materials.add_material(Box::new(Refractive::new(Spectrum::new(0.9, 0.9, 0.9), 0.1, 1.2)));

    let mut hitables = HitableStore::new();
    hitables.push(Box::new(Sphere::new(
        TransformSequence::new(
            Vec3::new(0.0, -200.5, -1.0),
            Quat::default()),
        200.0,
        ground,
    )));
    hitables.push(Box::new(TracedSDF::new(
        sdfu::Sphere::new(0.45)
            .subtract(
                sdfu::Box::new(Vec3::new(0.25, 0.25, 1.5)))
            .union_smooth(
                sdfu::Sphere::new(0.3).translate(Vec3::new(0.3, 0.3, 0.0)),
                0.1)
            .union_smooth(
                sdfu::Sphere::new(0.3).translate(Vec3::new(-0.3, 0.3, 0.0)),
                0.1)
            .subtract(
                sdfu::Box::new(Vec3::new(0.2, 2.0, 0.2)))
            .translate(Vec3::new(-0.2, 0.0, -1.0)),
        pink_diffuse,
    )));
    hitables.push(Box::new(Sphere::new(
        TransformSequence::new(
            Vec3::new(-0.2, -0.1, -1.0),
            Quat::default()),
        0.15,
        silver,
    )));
    hitables.push(Box::new(Sphere::new(
        TransformSequence::new(
            Vec3::new(1.0, -0.25, -1.0),
            Quat::default()),
        0.25,
        gold,
    )));
    hitables.push(Box::new(Sphere::new(
        TransformSequence::new(
            |t: f32| -> Vec3 {
                Vec3::new(
                    0.2 - (2.0 * t * std::f32::consts::PI).sin() * 0.15,
                    -0.4,
                    -0.35 - (2.0 * t * std::f32::consts::PI).cos() * 0.15)
            },
            Quat::default()),
        0.1,
        glass,
    )));
    hitables.push(Box::new(Sphere::new(
        TransformSequence::new(
            Vec3::new(-0.25, -0.375, -0.15),
            Quat::default()),
        0.125,
        glass_rough,
    )));
    hitables.push(Box::new(Sphere::new(
        TransformSequence::new(
            Vec3::new(-0.6, -0.375, -0.5),
            // |t: f32| -> Vec3 {
            //     Vec3::new(
            //         -0.5 + (2.0 * t * std::f32::consts::PI).cos() * 1.5,
            //         -0.375,
            //         -0.5 - (2.0 * t * std::f32::consts::PI).sin() * 1.5)
            // },
            Quat::default()),
        0.125,
        gold_rough,
    )));

    World {
        materials,
        hitables,
    }
}

fn compute_luminance(world: &World, mut ray: Ray, time: f32, rng: &mut ThreadRng) -> Spectrum {
    let mut luminance = Spectrum::zero();
    let mut throughput = Spectrum::one();
    for bounce in 0.. {
        if let Some(intersection) = world.hitables.hit(&ray, time, 0.001..1000.0) {
            let material = world.materials.get(intersection.material);

            let inv_ray_dir = -*ray.dir();

            luminance += material.le(inv_ray_dir) * throughput;

            let scatter = material.scatter(ray, intersection, rng);

            if let Some(se) = scatter {
                if se.pdf == 0.0 || se.f.is_black() {
                    break;
                }
                throughput *= se.f / se.pdf * se.wi.dot(intersection.normal).abs();
                ray = Ray::new(intersection.point, se.wi);
            } else {
                break;
            }
        } else {
            let dir = ray.dir();
            let t = 0.5 * (dir.y + 1.0);

            luminance += throughput * Spectrum(Vec3::lerp(Vec3::one(), Vec3::new(0.5, 0.7, 1.0), t));
            break;
        }

        if bounce >= MAX_BOUNCES {
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

    luminance
}

fn main() {
    let world = setup();

    let mut img = image::RgbImage::new(DIMS.0, DIMS.1);

    let camera_position_sequence: Sequence<[f32; 3]> = Sequence::new(
        vec![0.0, 1.0, 2.0, 4.0, 5.0],
        vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [-0.5, 0.0, 2.0], [0.5, 0.0, 2.0], [0.0, 0.0, 1.0]],
        minterpolate::InterpolationFunction::Linear,
        false,
    );
    let camera_transform_sequence = TransformSequence::new(camera_position_sequence, Quat::default());
    let camera = Arc::new(PinholeCamera::new(DIMS.0 as f32 / DIMS.1 as f32, camera_transform_sequence));

    let mut pixels = vec![Spectrum::zero(); DIMS.0 as usize * DIMS.1 as usize];

    let frame_rate = 24;
    let frame_range = 3..4;
    let shutter_speed = 1.0 / 24.0;

    for frame in frame_range {
        let mutated = AtomicUsize::new(0);
        let start = Instant::now();

        let frame_start = frame as f32 * (1.0 / frame_rate as f32);
        let frame_end = frame_start + shutter_speed;

        pixels.par_iter_mut().enumerate().for_each(|(i, p)| {
            let x = i % DIMS.0 as usize;
            let y = (i - x) / DIMS.0 as usize;
            let col = (0..SAMPLES)
                .map(|_| {
                    let mut rng = thread_rng();
                    let uniform = Uniform::new(0.0, 1.0);
                    let (r1, r2) = (uniform.sample(&mut rng), uniform.sample(&mut rng));
                    let uv = Vec2::new(
                        (x as f32 + r1) / DIMS.0 as f32,
                        (y as f32 + r2) / DIMS.1 as f32,
                    );
                    let time = rng.gen_range(frame_start, frame_end);

                    let ray = camera.clone().get_ray(uv, time);
                    compute_luminance(&world, ray, time, &mut rng)
                })
                .fold(Spectrum::zero(), |a, b| a + b);
            let col = col / SAMPLES as f32;
            *p = col;
            if i % 100_000 == 0 {
                let n = mutated.fetch_add(1, Ordering::Relaxed);
                println!(
                    "{}% finished...",
                    (n as f32 / (DIMS.0 * DIMS.1) as f32 * 100.0 * 100_000.0).round() as u32
                );
            }
        });

        let time = Instant::now() - start;
        let time_secs = time.as_secs();
        let time_millis = time.subsec_millis();

        println!(
            "Done in {} seconds.",
            time_secs as f32 + time_millis as f32 / 1000.0
        );

        println!("Post processing image...");

        for (x, y, pixel) in img.enumerate_pixels_mut() {
            let idx = x + (DIMS.1 - 1 - y) * DIMS.0;
            *pixel = (pixels[idx as usize]).gamma_correct(2.2).into();
        }

        let args: Vec<String> = std::env::args().collect();
        let default = String::from(format!("renders/frame{}.png", frame));
        let filename = args.get(1).unwrap_or(&default);
        println!("Saving to {}...", filename);

        img.save(filename).unwrap();
    }
    drop(world);
}
