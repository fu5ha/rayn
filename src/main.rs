use generic_array::typenum::*;

mod animation;
mod camera;
mod film;
mod filter;
mod hitable;
mod integrator;
mod light;
mod material;
mod math;
mod ray;
mod sampler;
mod sdf;
mod spectrum;
mod sphere;
mod world;

use camera::{CameraHandle, CameraStore, ThinLensCamera};
use film::{ChannelKind, Film};
use filter::BlackmanHarrisFilter;
use hitable::HitableStore;
use integrator::PathTracingIntegrator;
use light::{Light, SphereLight};
use material::{Dielectric, MaterialStore, Sky};
use math::{Extent2u, Vec3};
use sdf::{BoxFold, MandelBox, SphereFold, TracedSDF};
use spectrum::Srgb;
use sphere::Sphere;
use world::World;

use std::time::Instant;

const RES: (usize, usize) = (1280, 720);
const SAMPLES: usize = 8;

const MB_ITERS: usize = 20;

fn setup() -> (CameraHandle, World) {
    let mut materials = MaterialStore::new();

    let grey = materials.add_material(Dielectric::new_remap(
        Srgb::new(0.3, 0.3, 0.3),
        0.35,
    ));

    let sky = materials.add_material(Sky {});

    let mut hitables = HitableStore::new();

    hitables.push(Sphere::new(Vec3::new(0.0, 0.0, 0.0), 300.0, sky));

    hitables.push(TracedSDF::new(
        MandelBox::new(MB_ITERS, BoxFold::new(1.0), SphereFold::new(0.5, 1.0), -2.0),
        grey,
    ));

    let mut lights: Vec<Box<dyn Light>> = Vec::new();

    let sun = Srgb::new(4.0, 3.0, 2.5) * 25000.0;
    let blue = Srgb::new(1.5, 2.5, 5.0) * 5.0;
    let pink = Srgb::new(4.5, 2.0, 3.0) * 5.0;

    lights.push(Box::new(SphereLight::new(
        Vec3::new(2.65, 3.0, -1.0) * 50.0,
        1.0,
        sun,
    )));

    // lights.push(Box::new(SphereLight::new(
    //     Vec3::new(-1.0, 2.0, 2.65) * 50.0,
    //     1.0,
    //     sun,
    // )));

    lights.push(Box::new(SphereLight::new(
        Vec3::new(1.5, -0.35, 1.95),
        0.1,
        blue,
    )));

    lights.push(Box::new(SphereLight::new(
        Vec3::new(1.5, 0.35, 1.95),
        0.1,
        pink,
    )));

    lights.push(Box::new(SphereLight::new(
        Vec3::new(-1.5, -0.35, 1.95),
        0.1,
        blue,
    )));

    lights.push(Box::new(SphereLight::new(
        Vec3::new(-1.5, 0.35, 1.95),
        0.1,
        pink,
    )));

    // 1
    let camera = ThinLensCamera::new(
        RES.0 as f32 / RES.1 as f32,
        60.0,
        0.005,
        Vec3::new(7.5, -2.0, 9.5) * 0.3,
        Vec3::new(0.0, 1.0, 1.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.5, -0.2, 2.0),
    );

    let mut cameras = CameraStore::new();

    let camera = cameras.add_camera(Box::new(camera));

    (
        camera,
        World {
            materials,
            hitables,
            lights,
            cameras,
        },
    )
}

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .unwrap();

    let (camera, world) = setup();

    let mut film = Film::<U4>::new(
        &[
            ChannelKind::Color,
            ChannelKind::Alpha,
            ChannelKind::Background,
            ChannelKind::WorldNormal,
        ],
        Extent2u::new(RES.0, RES.1),
    )
    .unwrap();

    let frame_rate = 24;
    let frame_range = 0..1;
    let shutter_speed = 1.0 / 24.0;

    let filter = BlackmanHarrisFilter::new(1.5);
    // let filter = BoxFilter::default();
    let integrator = PathTracingIntegrator { max_bounces: 5 };

    for frame in frame_range {
        let start = Instant::now();

        let frame_start = frame as f32 * (1.0 / frame_rate as f32);
        let frame_end = frame_start + shutter_speed;

        film.render_frame_into(
            &world,
            camera,
            &integrator,
            &filter,
            Extent2u::new(16, 16),
            frame_start..frame_end,
            SAMPLES,
        );

        let time = Instant::now() - start;
        let time_secs = time.as_secs();
        let time_millis = time.subsec_millis();

        println!(
            "Done in {} seconds.",
            time_secs as f32 + time_millis as f32 / 1000.0
        );

        println!("Post processing image...");

        film.save_to(
            &[ChannelKind::WorldNormal, ChannelKind::Color],
            "renders",
            format!("mb_{}_iters_frame_{}", MB_ITERS, frame),
            true,
        )
        .unwrap();
    }
}
