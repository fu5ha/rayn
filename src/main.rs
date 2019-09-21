use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use rand::distributions::Uniform;
use rand::prelude::*;
use rayon::prelude::*;

mod animation;
mod camera;
mod color;
mod hitable;
mod material;
mod math;
mod ray;
mod sphere;

use crate::animation::{ Sequence, TransformSequence };
use crate::camera::{ Camera, PinholeCamera };
use crate::color::Color;
use crate::hitable::{Hitable, HitableList};
use crate::material::{Dielectric, Metal, Refractive};
use crate::math::{ Vec2, Vec3, Quat };
use crate::ray::Ray;
use crate::sphere::Sphere;

use lazy_static::lazy_static;
lazy_static! {
    static ref PINK_DIFFUSE: Dielectric = Dielectric::new(Color::new(0.7, 0.3, 0.4), 0.0);
    static ref GROUND: Dielectric = Dielectric::new(Color::new(0.35, 0.3, 0.45), 0.0);
    static ref GOLD: Metal = Metal::new(Color::new(1.0, 0.9, 0.5), 0.0);
    static ref GOLD_ROUGH: Metal = Metal::new(Color::new(1.0, 0.9, 0.5), 0.2);
    static ref SILVER: Metal = Metal::new(Color::new(0.9, 0.9, 0.9), 0.05);
    static ref GLASS: Refractive = Refractive::new(Color::new(0.9, 0.9, 0.9), 0.0, 1.5);
    static ref GLASS_ROUGH: Refractive = Refractive::new(Color::new(0.9, 0.9, 0.9), 0.2, 1.5);
    static ref WORLD: HitableList = {
        let mut world = HitableList::new();
        world.push(Box::new(Sphere::new(
            TransformSequence::new(
                Vec3::new(0.0, -200.5, -1.0),
                Quat::default()),
            200.0,
            &*GROUND,
        )));
        world.push(Box::new(Sphere::new(
            TransformSequence::new(
                Vec3::new(0.0, 0.0, -1.0),
                Quat::default()),
            0.5,
            &*SILVER,
        )));
        world.push(Box::new(Sphere::new(
            TransformSequence::new(
                Vec3::new(-1.0, 0.0, -1.0),
                Quat::default()),
            0.5,
            &*PINK_DIFFUSE,
        )));
        world.push(Box::new(Sphere::new(
            TransformSequence::new(
                Vec3::new(1.0, -0.25, -1.0),
                Quat::default()),
            0.25,
            &*GOLD,
        )));
        world.push(Box::new(Sphere::new(
            TransformSequence::new(
                |t: f32| -> Vec3 {
                    Vec3::new(
                        0.4 - (2.0 * t * std::f32::consts::PI).cos() * 0.3,
                        -0.375,
                        -0.5 + (2.0 * t * std::f32::consts::PI).sin() * 0.3)
                },
                Quat::default()),
            0.125,
            &*GLASS,
        )));
        world.push(Box::new(Sphere::new(
            TransformSequence::new(
                |t: f32| -> Vec3 {
                    Vec3::new(
                        0.2 - (2.0 * t * std::f32::consts::PI).sin() * 0.15,
                        -0.4,
                        -0.35 - (2.0 * t * std::f32::consts::PI).cos() * 0.15)
                },
                Quat::default()),
            0.1,
            &*GLASS,
        )));
        world.push(Box::new(Sphere::new(
            TransformSequence::new(
                Vec3::new(-0.25, -0.375, -0.15),
                Quat::default()),
            0.125,
            &*GLASS_ROUGH,
        )));
        world.push(Box::new(Sphere::new(
            TransformSequence::new(
                |t: f32| -> Vec3 {
                    Vec3::new(
                        -0.5 + (2.0 * t * std::f32::consts::PI).cos() * 1.5,
                        -0.375,
                        -0.5 + (2.0 * t * std::f32::consts::PI).sin() * 1.5)
                },
                Quat::default()),
            0.125,
            &*GOLD_ROUGH,
        )));
        world
    };
}

const DIMS: (u32, u32) = (1920, 1080);
const SAMPLES: usize = 128;
const MAX_BOUNCES: usize = 256;

fn compute_color(ray: Ray, time: f32, rng: &mut ThreadRng, bounces: usize) -> Color {
    if bounces < MAX_BOUNCES {
        if let Some(record) = WORLD.hit(&ray, time, 0.001..1000.0) {
            let scatter = record.material.scatter(&ray, &record, rng);
            if let Some(scattering_event) = scatter {
                compute_color(Ray::new(record.point, scattering_event.out_dir, scattering_event.out_ior), time, rng, bounces + 1) 
                    * scattering_event.attenuation
                    + scattering_event.emission
            } else {
                Color::zero()
            }
        } else {
            let dir = ray.dir();
            let t = 0.5 * (dir.y + 1.0);

            Color(Vec3::lerp(Vec3::one(), Vec3::new(0.5, 0.7, 1.0), t))
        }
    } else {
        Color::zero()
    }
}

fn main() {
    let mut img = image::RgbImage::new(DIMS.0, DIMS.1);

    let camera_position_sequence: Sequence<[f32; 3]> = Sequence::new(
        vec![0.0, 1.0, 2.0, 4.0, 5.0],
        vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [-0.5, 0.0, 2.0], [0.5, 0.0, 2.0], [0.0, 0.0, 1.0]],
        minterpolate::InterpolationFunction::Linear,
        false,
    );
    let camera_transform_sequence = TransformSequence::new(camera_position_sequence, Quat::default());
    let camera = Arc::new(PinholeCamera::new(DIMS.0 as f32 / DIMS.1 as f32, camera_transform_sequence));

    let mut pixels = vec![Color::zero(); DIMS.0 as usize * DIMS.1 as usize];

    let frame_rate = 24;
    let frame_range = 0..144;
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
                    compute_color(ray, time, &mut rng, 0)
                })
                .fold(Color::zero(), |a, b| a + b);
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

        for (x, y, pixel) in img.enumerate_pixels_mut() {
            let idx = x + (DIMS.1 - 1 - y) * DIMS.0;
            *pixel = pixels[idx as usize].gamma_correct(2.2).into();
        }

        let time = Instant::now() - start;
        let time_secs = time.as_secs();
        let time_millis = time.subsec_millis();

        println!(
            "Done in {} seconds.",
            time_secs as f32 + time_millis as f32 / 1000.0
        );
        let args: Vec<String> = std::env::args().collect();
        let default = String::from(format!("renders/frame{}.png", frame));
        let filename = args.get(1).unwrap_or(&default);
        println!("Saving to {}", filename);

        img.save(filename).unwrap();
    }
}
