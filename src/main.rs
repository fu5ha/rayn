use generic_array::typenum::*;
use sdfu::SDF;

mod animation;
mod camera;
mod film;
mod filter;
mod hitable;
mod integrator;
mod material;
mod math;
mod ray;
mod sdf;
mod spectrum;
mod sphere;
mod world;

use camera::{CameraHandle, CameraStore, ThinLensCamera};
use film::{ChannelKind, Film};
use filter::BlackmanHarrisFilter;
use hitable::HitableStore;
use integrator::PathTracingIntegrator;
use material::{Dielectric, Emissive, MaterialStore, Sky};
use math::{f32x4, Extent2u, Vec3};
use sdf::{BoxFold, MandelBox, SphereFold, TracedSDF};
use spectrum::WSrgb;
use sphere::Sphere;
use world::World;

use std::time::Instant;

const RES: (usize, usize) = (960, 540);
const SAMPLES: usize = 16;

const MB_ITERS: usize = 20;

fn setup() -> (CameraHandle, World) {
    let mut materials = MaterialStore::new();

    let pink = materials.add_material(Dielectric::new(
        WSrgb::new_splat(0.75, 0.5, 0.55),
        f32x4::from(0.1),
    ));

    let white_emissive = materials.add_material(Emissive::new(WSrgb::new_splat(2.0, 3.0, 4.5)));

    let sky = materials.add_material(Sky {});

    let mut hitables = HitableStore::new();

    hitables.push(Sphere::new(Vec3::new(0.0, 0.0, 0.0), 300.0, sky));

    hitables.push(TracedSDF::new(
        MandelBox::new(MB_ITERS, BoxFold::new(1.0), SphereFold::new(0.5, 1.0), 2.0),
        pink,
    ));

    hitables.push(Sphere::new(
        Vec3::new(3.25, 1.75, 5.25),
        0.1,
        white_emissive,
    ));

    let camera = ThinLensCamera::new(
        RES.0 as f32 / RES.1 as f32,
        60.0,
        0.0001,
        Vec3::new(3.5, 1.875, 6.05),
        Vec3::new(0.0, 0.875, 0.9),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(3.0, 1.875, 6.0),
    );

    // let camera = ThinLensCamera::new(
    //     RES.0 as f32 / RES.1 as f32,
    //     60.0,
    //     0.005,
    //     Vec3::new(1.8, 0.0, 2.25),
    //     Vec3::new(1.5, 0.0, 1.5),
    //     Vec3::new(0.0, 1.0, 0.0),
    //     Vec3::new(1.8, 0.0, 2.0),
    // );

    // let camera = ThinLensCamera::new(
    //     RES.0 as f32 / RES.1 as f32,
    //     60.0,
    //     0.015,
    //     Vec3::new(1.5, 1.5, 2.5),
    //     Vec3::new(1.7, 1.7, 1.9),
    //     Vec3::new(0.0, 1.0, 0.0),
    //     Vec3::new(1.8, 1.8, 2.3),
    // );

    let mut cameras = CameraStore::new();

    let camera = cameras.add_camera(Box::new(camera));

    (
        camera,
        World {
            materials,
            hitables,
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

    let mut film = Film::<U3>::new(
        &[
            ChannelKind::Color,
            ChannelKind::Background,
            ChannelKind::WorldNormal,
        ],
        Extent2u::new(RES.0, RES.1),
    )
    .unwrap();

    let frame_rate = 24;
    let frame_range = 0..1;
    let shutter_speed = 1.0 / 24.0;

    let filter = BlackmanHarrisFilter::new(2.0);
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
            Extent2u::new(8, 8),
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
            false,
        )
        .unwrap();
    }
}
