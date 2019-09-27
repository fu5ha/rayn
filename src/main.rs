use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use dynamic_arena::{ DynamicArena, NonSend };

use rand::distributions::Uniform;
use rand::prelude::*;
use rayon::prelude::*;

use sdfu::SDF;

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
use camera::{ ThinLensCamera };
use spectrum::{ IsSpectrum, RGBSpectrum };
use hitable::{Hitable, HitableStore};
use material::{Dielectric, Emissive, Checkerboard3d, MaterialStore, Metal, Refractive};
use math::{ Vec2, Vec3, Quat };
use ray::Ray;
use sdf::TracedSDF;
use sphere::Sphere;
use world::World;

const DIMS: (u32, u32) = (1280, 720);
const CHUNK_SIZE: usize = 16 * 16;
const SAMPLES: usize = 256;
const MAX_BOUNCES: usize = 5;

type Spectrum = RGBSpectrum;
const NUM_PIXELS: usize = (DIMS.0 * DIMS.1) as usize;

fn setup() -> World<Spectrum> {
    let white_emissive = Emissive::new(Spectrum::new(1.0, 1.0, 1.5), Dielectric::new(Spectrum::new(0.5, 0.5, 0.5), 0.0));
    let checkerboard = Checkerboard3d::new(
        Vec3::new(0.15, 0.15, 0.15),
        Dielectric::new(Spectrum::new(0.9, 0.35, 0.55), 0.0),
        Refractive::new(Spectrum::new(0.9, 0.9, 0.9), 0.0, 1.5));

    let mut materials = MaterialStore::new();
    let checkerboard = materials.add_material(Box::new(checkerboard));
    let white_emissive = materials.add_material(Box::new(white_emissive));
    let ground = materials.add_material(Box::new(Dielectric::new(Spectrum::new(0.25, 0.2, 0.35), 0.3)));
    let gold = materials.add_material(Box::new(Metal::new(Spectrum::new(1.0, 0.9, 0.5), 0.0)));
    let silver = materials.add_material(Box::new(Metal::new(Spectrum::new(0.9, 0.9, 0.9), 0.05)));
    let gold_rough = materials.add_material(Box::new(Metal::new(Spectrum::new(1.0, 0.9, 0.5), 0.5)));
    let glass = materials.add_material(Box::new(Refractive::new(Spectrum::new(0.9, 0.9, 0.9), 0.0, 1.5)));
    let glass_rough = materials.add_material(Box::new(Refractive::new(Spectrum::new(0.9, 0.9, 0.9), 0.5, 1.5)));

    let mut hitables = HitableStore::<Spectrum>::new();
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
                sdfu::Box::new(Vec3::new(0.125, 0.125, 1.5)).translate(Vec3::new(-0.3, 0.3, 0.0)))
            .subtract(
                sdfu::Box::new(Vec3::new(0.125, 0.125, 1.5)).translate(Vec3::new(0.3, 0.3, 0.0)))
            .subtract(
                sdfu::Box::new(Vec3::new(1.5, 0.1, 0.1)).translate(Vec3::new(0.0, 0.3, 0.0)))
            .subtract(
                sdfu::Box::new(Vec3::new(0.2, 2.0, 0.2)))
            .translate(Vec3::new(-0.2, 0.0, -1.0)),
        checkerboard,
    )));
    hitables.push(Box::new(Sphere::new(
        TransformSequence::new(
            Vec3::new(-0.2, -0.1, -1.0),
            Quat::default()),
        0.15,
        white_emissive,
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
            Vec3::new(-1.5, 0.0, -1.0),
            Quat::default()),
        0.4,
        silver,
    )));
    hitables.push(Box::new(Sphere::new(
        TransformSequence::new(
            |t: f32| -> Vec3 {
                Vec3::new(
                    0.2 - (t * std::f32::consts::PI).sin() * 0.15,
                    -0.4,
                    -0.35 - (t * std::f32::consts::PI).cos() * 0.15)
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
            |t: f32| -> Vec3 {
                Vec3::new(
                    -0.5 + (2.0 * t * std::f32::consts::PI).cos() * 1.5,
                    -0.375,
                    -0.5 - (2.0 * t * std::f32::consts::PI).sin() * 1.5)
            },
            Quat::default()),
        0.125,
        gold_rough,
    )));

    let camera_position_sequence: Sequence<[f32; 3]> = Sequence::new(
        vec![0.0, 1.0, 3.0, 4.0, 5.0, 8.0, 9.0],
        vec![
            [0.0, 0.0, 1.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.5],
            [0.1, 0.3, -0.5],
            [0.1, 0.3, -1.5],
            [-0.2, 0.3, -2.0],
            [-0.2, -0.3, -0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.5],
        ],
        minterpolate::InterpolationFunction::CatmullRomSpline,
        false,
    );

    let camera_lookat_sequence: Sequence<[f32; 3]> = Sequence::new(
        vec![0.0, 1.0, 3.0, 3.5, 5.0],
        vec![
            [-0.2, 0.1, -1.0],
            [-0.2, 0.1, -1.0],
            [-0.2, 0.1, -1.0],
            [0.1, 0.3, -1.0],
            [0.1, 0.3, -2.0],
            [-0.2, 0.1, -1.0],
            [-0.2, 0.1, -1.0],
        ],
        minterpolate::InterpolationFunction::CatmullRomSpline,
        false,
    );

    let camera_focus_sequence: Sequence<[f32; 3]> = Sequence::new(
        vec![0.0, 1.0, 3.0, 4.0, 5.0, 8.0],
        vec![
            [-0.2, 0.1, -1.0],
            [-0.2, 0.1, -1.0],
            [-0.2, 0.1, -1.0],
            [0.1, 0.3, -1.0],
            [0.1, 0.3, -1.5],
            [-0.2, 0.1, -1.0],
            [-0.2, 0.1, -1.0],
            [-0.2, 0.1, -1.0],
        ],
        minterpolate::InterpolationFunction::CatmullRomSpline,
        false,
    );
    let camera_aperture_sequence: Sequence<f32> = Sequence::new(
        vec![0.0, 1.0, 3.0, 4.0, 5.0, 8.0],
        vec![
            0.0225,
            0.0225,
            0.0225,
            0.0125,
            0.0125,
            0.0225,
            0.0125,
            0.0125,
        ],
        minterpolate::InterpolationFunction::CatmullRomSpline,
        false,
    );
    let camera = ThinLensCamera::new(
        DIMS.0 as f32 / DIMS.1 as f32,
        60.0,
        camera_aperture_sequence,
        camera_position_sequence,
        camera_lookat_sequence,
        Vec3::new(0.0, 1.0, 0.0),
        camera_focus_sequence);


    World {
        materials,
        hitables,
        camera: Box::new(camera),
    }
}

fn compute_luminance(world: &World<Spectrum>, mut ray: Ray, time: f32, rng: &mut ThreadRng) -> Spectrum {
    let mut luminance = Spectrum::zero();
    let mut throughput = Spectrum::one();
    let arena = DynamicArena::<'_, NonSend>::new_bounded();
    for bounce in 0.. {
        if let Some(mut intersection) = world.hitables.hit(&ray, time, 0.001..1000.0) {
            let wi = *ray.dir();
            let material = world.materials.get(intersection.material);

            material.setup_scattering_functions(&mut intersection, &arena);
            let bsdf = unsafe { intersection.bsdf.assume_init() };

            luminance += bsdf.le(-wi, &mut intersection) * throughput;

            let scattering_event = bsdf.scatter(wi, &mut intersection, rng);

            if let Some(se) = scattering_event {
                if se.pdf == 0.0 || se.f.is_black() {
                    break;
                }
                throughput *= se.f / se.pdf * se.wi.dot(intersection.normal).abs();
                ray = intersection.create_ray(se.wi);
            } else {
                break;
            }
        } else {
            let dir = ray.dir();
            let t = 0.5 * (dir.y + 1.0);

            luminance += throughput * RGBSpectrum(Vec3::lerp(Vec3::one(), Vec3::new(0.5, 0.7, 1.0), t));
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
    rayon::ThreadPoolBuilder::new().num_threads(num_cpus::get()).build_global().unwrap();

    let world = setup();

    let mut img = image::RgbImage::new(DIMS.0, DIMS.1);

    let mut pixels = Vec::with_capacity(DIMS.0 as usize * DIMS.1 as usize);
    for i in 0..NUM_PIXELS {
        let x = i % DIMS.0 as usize;
        let y = (i - x) / DIMS.0 as usize;
        pixels.push(((x, y), Spectrum::zero()));
    }

    let frame_rate = 24;
    let frame_range = 0..216;
    let shutter_speed = 1.0 / 24.0;

    for frame in frame_range {
        let mutated = AtomicUsize::new(0);
        let start = Instant::now();

        let frame_start = frame as f32 * (1.0 / frame_rate as f32);
        let frame_end = frame_start + shutter_speed;

        pixels.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
            let mut rng = thread_rng();
            for ((x, y), pixel) in chunk.iter_mut() {
                let col: Spectrum = (0..SAMPLES)
                    .map(|_| {
                        let uniform = Uniform::new(0.0, 1.0);
                        let (r1, r2) = (uniform.sample(&mut rng), uniform.sample(&mut rng));
                        let uv = Vec2::new(
                            (*x as f32 + r1) / DIMS.0 as f32,
                            (*y as f32 + r2) / DIMS.1 as f32,
                        );
                        let time = rng.gen_range(frame_start, frame_end);

                        let ray = world.camera.get_ray(uv, time, &mut rng);
                        compute_luminance(&world, ray, time, &mut rng)
                    })
                    .sum();
                let col = col / SAMPLES as f32;
                *pixel = col;
                if (*x + *y * DIMS.0 as usize) % 50_000 == 0 {
                    let n = mutated.fetch_add(1, Ordering::Relaxed);
                    println!(
                        "{}% finished...",
                        (n as f32 / (DIMS.0 * DIMS.1) as f32 * 100.0 * 50_000.0).round() as u32
                    );
                }
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
            *pixel = (pixels[idx as usize].1).gamma_correct(2.2).into();
        }

        let args: Vec<String> = std::env::args().collect();
        let default = String::from(format!("renders/frame{}.png", frame));
        let filename = args.get(1).unwrap_or(&default);
        println!("Saving to {}...", filename);

        img.save(filename).unwrap();
    }
    drop(world);
}
