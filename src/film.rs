use dynamic_arena::{DynamicArena, NonSend};
use generic_array::{ArrayLength, GenericArray};

use rand::distributions::Uniform;
use rand::prelude::*;

use crate::camera::CameraHandle;
use crate::filter::Filter;
use crate::hitable::Intersection;
use crate::integrator::Integrator;
use crate::math::{Aabr, Aabri, Aabru, Extent2u, Vec2, Vec2i, Vec2u, Vec3};
use crate::spectrum::{IsSpectrum, Rgb, Xyz};
use crate::world::World;

use std::collections::HashMap;
use std::ops::Range;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

macro_rules! declare_channels {
    {
        $($name:ident => {
            storage: $storage:ident,
            init: $initialize:expr,
        }),+
    } => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum ChannelKind {
            $($name,)+
        }

        pub enum ChannelTileStorage {
            $($name(Vec<($storage, f32)>),)+
        }

        impl ChannelTileStorage {
            fn new(kind: ChannelKind, res: Extent2u) -> Self {
                use ChannelKind::*;
                match kind {
                    $($name => Self::$name(vec![($initialize, 0.0); res.w * res.h]),)+
                }
            }
        }

        pub enum ChannelStorage {
            $($name(Vec<$storage>),)+
        }

        impl ChannelStorage {
            fn new(kind: ChannelKind, res: Extent2u) -> Self {
                use ChannelKind::*;
                match kind {
                    $($name => Self::$name(vec![$initialize; res.w * res.h]),)+
                }
            }

            pub fn kind(&self) -> ChannelKind {
                match *self {
                    $( ChannelStorage::$name(_) => ChannelKind::$name, )+
                }
            }

            pub fn copy_from_tile(&mut self, other: &ChannelTileStorage, full_res: Extent2u, tile_bounds: Aabru) -> Result<(), ()> {
                let extent = tile_bounds.size();
                match (self, other) {
                    $( (ChannelStorage::$name(this_buf), ChannelTileStorage::$name(tile_buf)) => {
                        for x in 0..extent.w {
                            for y in 0..extent.h {
                                let tile_idx = x + y * extent.w;
                                let this_idx = (tile_bounds.min.x + x) + (tile_bounds.min.y + y) * full_res.w;
                                let tile = tile_buf[tile_idx];
                                this_buf[this_idx] = tile.0 / tile.1;
                                // if let ChannelTileStorage::Color(tile_buf) = other {
                                //     let tile = tile_buf[tile_idx];
                                //     let col = tile.0 / tile.1;
                                //     if col.x <= 0.001 || col.y <= 0.001 || col.z <= 0.001 {
                                //         println!("{}, {}, {}", col.x, col.y, col.z);
                                //     }
                                // }
                            }
                        }
                        Ok(())
                    }, )+
                    _ => Err(())
                }
            }
        }
    }
}

declare_channels! {
    Color => {
        storage: Xyz,
        init: Xyz::zero(),
    },
    Alpha => {
        storage: f32,
        init: 0f32,
    },
    Background => {
        storage: Xyz,
        init: Xyz::zero(),
    },
    WorldNormal => {
        storage: Vec3,
        init: Vec3::zero(),
    }
}

macro_rules! channel_storage_index {
    ($storage:expr, $channel:ident, $idx:expr) => {
        if let ChannelStorage::$channel(x) = &$storage[$idx] {
            x
        } else {
            panic!("Attempted to index into channel storage array with wrong channel type.");
        }
    };
}

pub struct Tile<N: ArrayLength<ChannelTileStorage>, F> {
    epoch: usize,
    channels: GenericArray<ChannelTileStorage, N>,
    raster_bounds: Aabru,
    sample_bounds: Aabri,
    screen_to_ndc_size: Vec2,
    filter_table: Vec<f32>,
    filter: F,
}

impl<N: ArrayLength<ChannelTileStorage>, F: Filter> Tile<N, F> {
    pub fn new<IC>(
        epoch: usize,
        channels: IC,
        res: Extent2u,
        raster_bounds: Aabru,
        filter: F,
    ) -> Self
    where
        IC: std::iter::ExactSizeIterator<Item = ChannelKind>,
    {
        let screen_to_ndc_size = Vec2::new(1.0 / res.w as f32, 1.0 / res.h as f32);

        let half_pixel = Vec2::new(0.5, 0.5);
        let filter_radius = filter.radius();

        let float_sample_bounds = Aabr {
            min: Vec2::new(
                raster_bounds.min.x as f32 - filter_radius,
                raster_bounds.min.y as f32 - filter_radius,
            ) + half_pixel,
            max: Vec2::new(
                raster_bounds.max.x as f32 + filter_radius,
                raster_bounds.max.y as f32 + filter_radius,
            ) + half_pixel,
        };

        let mut filter_table = Vec::new();

        for offset in 0..filter_radius.ceil() as usize + 1 {
            filter_table.push(filter.evaluate(offset as f32));
        }

        Tile {
            epoch,
            channels: GenericArray::from_exact_iter(
                channels.map(|kind| ChannelTileStorage::new(kind, raster_bounds.size())),
            )
            .expect("Incorrect number of channels passed to tile creation"),
            raster_bounds,
            sample_bounds: Aabri {
                min: Vec2i::new(
                    float_sample_bounds.min.x.floor() as isize,
                    float_sample_bounds.min.y.floor() as isize,
                ),
                max: Vec2i::new(
                    float_sample_bounds.max.x.ceil() as isize,
                    float_sample_bounds.max.y.ceil() as isize,
                ),
            },
            screen_to_ndc_size,
            filter_table,
            filter,
        }
    }

    pub fn add_sample(
        &mut self,
        sample_screen_coord: Vec2,
        spect: Xyz,
        first_intersection: Option<Intersection>,
    ) {
        let p_discrete = sample_screen_coord - Vec2::new(0.5, 0.5);
        let filter_radius = self.filter.radius();
        let sample_influence_bounds = Aabru {
            min: Vec2u::new(
                ((p_discrete.x - filter_radius).ceil() as isize)
                    .max(self.raster_bounds.min.x as isize) as usize,
                ((p_discrete.y - filter_radius).ceil() as isize)
                    .max(self.raster_bounds.min.y as isize) as usize,
            ),
            max: Vec2u::new(
                ((p_discrete.x + filter_radius).ceil() as isize)
                    .min(self.raster_bounds.max.x as isize) as usize,
                ((p_discrete.y + filter_radius).ceil() as isize)
                    .min(self.raster_bounds.max.y as isize) as usize,
            ),
        };

        let width = self.raster_bounds.size().w;
        let p_raster = Vec2i::new(
            sample_screen_coord.x.floor() as isize,
            sample_screen_coord.y.floor() as isize,
        );

        for x in sample_influence_bounds.min.x..sample_influence_bounds.max.x {
            for y in sample_influence_bounds.min.y..sample_influence_bounds.max.y {
                let weight = self.filter_table[(x as isize - p_raster.x).abs() as usize]
                    * self.filter_table[(y as isize - p_raster.y).abs() as usize];

                let tile_pixel = Vec2u::new(x, y) - self.raster_bounds.min;
                let idx = tile_pixel.x + tile_pixel.y * width;

                for channel in self.channels.iter_mut() {
                    match channel {
                        ChannelTileStorage::Color(ref mut buf) => {
                            let (col_sum, weight_sum) = &mut buf[idx];
                            if let Some(_) = first_intersection {
                                *col_sum += spect * weight;
                            }
                            *weight_sum += weight;
                        }
                        ChannelTileStorage::Background(ref mut buf) => {
                            let (col_sum, weight_sum) = &mut buf[idx];
                            if let None = first_intersection {
                                *col_sum += spect * weight;
                            }
                            *weight_sum += weight;
                        }
                        ChannelTileStorage::Alpha(ref mut buf) => {
                            let (col_sum, weight_sum) = &mut buf[idx];
                            if let Some(_) = first_intersection {
                                *col_sum += 1.0 * weight;
                            }
                            *weight_sum += weight;
                        }
                        ChannelTileStorage::WorldNormal(ref mut buf) => {
                            let (col_sum, weight_sum) = &mut buf[idx];
                            if let Some(isec) = first_intersection {
                                *col_sum += isec.normal * weight;
                            }
                            *weight_sum += weight;
                        }
                    }
                }
            }
        }
    }
}

pub struct Film<N: ArrayLength<ChannelStorage>> {
    channel_indices: HashMap<ChannelKind, usize>,
    channels: Mutex<GenericArray<ChannelStorage, N>>,
    progressive_epoch: usize,
    this_epoch_tiles_finished: AtomicUsize,
    res: Extent2u,
}

impl<'a, N: ArrayLength<ChannelStorage>> Film<N> {
    pub fn new(channels: &[ChannelKind], res: Extent2u) -> Result<Self, String> {
        let mut channel_indices = HashMap::new();
        for (i, kind) in channels.iter().enumerate() {
            if let Some(_) = channel_indices.insert(*kind, i) {
                return Err(String::from(format!(
                    "Attempted to create multiple {:?} channels",
                    *kind
                )));
            }
        }
        Ok(Film {
            channel_indices,
            channels: Mutex::new(
                GenericArray::from_exact_iter(
                    channels
                        .into_iter()
                        .map(|kind| ChannelStorage::new(*kind, res)),
                )
                .expect("Generic type length does not match the number of channels."),
            ),
            progressive_epoch: 0,
            this_epoch_tiles_finished: AtomicUsize::new(0),
            res,
        })
    }

    pub fn save_to<P: AsRef<std::path::Path>, IS: Into<String>>(
        &self,
        write_channels: &[ChannelKind],
        output_folder: P,
        base_name: IS,
        transparent_background: bool,
    ) -> Result<(), String> {
        let base_name = base_name.into();

        let channels = self.channels.lock().unwrap();

        for kind in write_channels.iter() {
            match *kind {
                ChannelKind::Color => {
                    let color_idx = self.channel_indices.get(&ChannelKind::Color);
                    let alpha_idx = self.channel_indices.get(&ChannelKind::Alpha);
                    let bg_idx = self.channel_indices.get(&ChannelKind::Background);

                    match (color_idx, alpha_idx, bg_idx, transparent_background) {
                        (Some(&color_idx), Some(&alpha_idx), _, true) => {
                            let color_buf = &channel_storage_index!(channels, Color, color_idx);
                            let alpha_buf = &channel_storage_index!(channels, Alpha, alpha_idx);
                            let mut img =
                                image::RgbaImage::new(self.res.w as u32, self.res.h as u32);
                            for (x, y, pixel) in img.enumerate_pixels_mut() {
                                let idx = x as usize + (self.res.h - 1 - y as usize) * self.res.w;
                                let col = color_buf[idx];
                                let a = alpha_buf[idx];
                                let rgb = Rgb::from(col).saturated().gamma_corrected(2.2) * a;
                                *pixel = image::Rgba([
                                    (rgb.r * 255.0).min(255.0).max(0.0) as u8,
                                    (rgb.g * 255.0).min(255.0).max(0.0) as u8,
                                    (rgb.b * 255.0).min(255.0).max(0.0) as u8,
                                    (a * 255.0).min(255.0).max(0.0) as u8,
                                ]);
                            }
                            let filename = output_folder
                                .as_ref()
                                .join(format!("{}_color.png", base_name.clone()));
                            println!("Saving to {}...", filename.display());
                            img.save(filename).unwrap();
                        }
                        (Some(&color_idx), _, Some(&bg_idx), false) => {
                            let color_buf = channel_storage_index!(channels, Color, color_idx);
                            let bg_buf = channel_storage_index!(channels, Background, bg_idx);
                            let mut img =
                                image::RgbImage::new(self.res.w as u32, self.res.h as u32);
                            for (x, y, pixel) in img.enumerate_pixels_mut() {
                                let i = x as usize + (self.res.h - 1 - y as usize) * self.res.w;
                                let col = color_buf[i];
                                let bg = bg_buf[i];
                                let rgb = Rgb::from(col + bg).saturated().gamma_corrected(2.2);
                                *pixel = image::Rgb([
                                    (rgb.r * 255.0).min(255.0).max(0.0) as u8,
                                    (rgb.g * 255.0).min(255.0).max(0.0) as u8,
                                    (rgb.b * 255.0).min(255.0).max(0.0) as u8,
                                ]);
                            }
                            let filename = output_folder
                                .as_ref()
                                .join(format!("{}_color.png", base_name.clone()));
                            println!("Saving to {}...", filename.display());
                            img.save(filename).unwrap();
                        }
                        (Some(&color_idx), _, None, false) => {
                            let color_buf = channel_storage_index!(channels, Color, color_idx);
                            let mut img =
                                image::RgbImage::new(self.res.w as u32, self.res.h as u32);
                            for (x, y, pixel) in img.enumerate_pixels_mut() {
                                let idx = x as usize + (self.res.h - 1 - y as usize) * self.res.w;
                                let rgb = Rgb::from(color_buf[idx]).gamma_corrected(2.2);
                                *pixel = image::Rgb([
                                    (rgb.r * 255.0).min(255.0).max(0.0) as u8,
                                    (rgb.g * 255.0).min(255.0).max(0.0) as u8,
                                    (rgb.b * 255.0).min(255.0).max(0.0) as u8,
                                ]);
                            }
                            let filename = output_folder
                                .as_ref()
                                .join(format!("{}_color.png", base_name.clone()));
                            println!("Saving to {}...", filename.display());
                            img.save(filename).unwrap();
                        }
                        _ => {
                            return Err(String::from(
                                "Attempted to write Color channel with insufficient channels",
                            ))
                        }
                    }
                }
                ChannelKind::Background => {
                    let idx =
                        *self
                            .channel_indices
                            .get(&ChannelKind::Background)
                            .ok_or(String::from(
                                "Attempted to write Background channel but it didn't exist",
                            ))?;
                    let buf = channel_storage_index!(channels, Background, idx);
                    let mut img = image::RgbImage::new(self.res.w as u32, self.res.h as u32);
                    for (x, y, pixel) in img.enumerate_pixels_mut() {
                        let idx = x as usize + (self.res.h - 1 - y as usize) * self.res.w;
                        let rgb = Rgb::from(buf[idx]).saturated().gamma_corrected(2.2);
                        *pixel = image::Rgb([
                            (rgb.r * 255.0).min(255.0).max(0.0) as u8,
                            (rgb.g * 255.0).min(255.0).max(0.0) as u8,
                            (rgb.b * 255.0).min(255.0).max(0.0) as u8,
                        ]);
                    }
                    let filename = output_folder
                        .as_ref()
                        .join(format!("{}_background.png", base_name.clone()));
                    println!("Saving to {}...", filename.display());
                    img.save(filename).unwrap();
                }
                ChannelKind::WorldNormal => {
                    let idx = *self.channel_indices.get(&ChannelKind::WorldNormal).ok_or(
                        String::from("Attempted to write WorldNormal channel but it didn't exist"),
                    )?;
                    let buf = channel_storage_index!(channels, WorldNormal, idx);
                    let mut img = image::RgbImage::new(self.res.w as u32, self.res.h as u32);
                    for (x, y, pixel) in img.enumerate_pixels_mut() {
                        let idx = x as usize + (self.res.h - 1 - y as usize) * self.res.w;
                        let vec = buf[idx];
                        let rgb = Rgb::from(vec * 0.5 + Vec3::new(0.5, 0.5, 0.5));
                        *pixel = image::Rgb([
                            (rgb.r * 255.0).min(255.0).max(0.0) as u8,
                            (rgb.g * 255.0).min(255.0).max(0.0) as u8,
                            (rgb.b * 255.0).min(255.0).max(0.0) as u8,
                        ]);
                    }
                    let filename = output_folder
                        .as_ref()
                        .join(format!("{}_normal.png", base_name.clone()));
                    println!("Saving to {}...", filename.display());
                    img.save(filename).unwrap();
                }
                ChannelKind::Alpha => {
                    let idx =
                        *self
                            .channel_indices
                            .get(&ChannelKind::Alpha)
                            .ok_or(String::from(
                                "Attempted to write Alpha channel but it didn't exist",
                            ))?;
                    let buf = channel_storage_index!(channels, Alpha, idx);
                    let mut img = image::GrayImage::new(self.res.w as u32, self.res.h as u32);
                    for (x, y, pixel) in img.enumerate_pixels_mut() {
                        let idx = x as usize + (self.res.h - 1 - y as usize) * self.res.w;
                        let a = buf[idx];
                        *pixel = image::Luma([(a * 255.0).min(255.0).max(0.0) as u8]);
                    }
                    let filename = output_folder
                        .as_ref()
                        .join(format!("{}_alpha.png", base_name.clone()));
                    println!("Saving to {}...", filename.display());
                    img.save(filename).unwrap();
                }
            }
        }
        Ok(())
    }
}

impl<'a, N: ArrayLength<ChannelStorage> + ArrayLength<ChannelTileStorage>> Film<N> {
    pub fn render_frame_into<I, S, F>(
        &'a mut self,
        world: &World<S>,
        camera: CameraHandle,
        integrator: I,
        filter: F,
        tile_size: Extent2u,
        time_range: Range<f32>,
        samples: usize,
    ) where
        F: Filter + Copy + Send,
        I: Integrator,
        S: IsSpectrum,
    {
        let camera = world.cameras.get(camera);
        let mut tiles = Vec::new();

        let rem = Vec2u::new((self.res.w) % tile_size.w, (self.res.h) % tile_size.h);
        {
            let channels = self.channels.lock().unwrap();
            for tile_x in 0..((self.res.w + rem.x) / tile_size.w) {
                for tile_y in 0..((self.res.h + rem.y) / tile_size.h) {
                    let start = Vec2u::new(tile_x * tile_size.w, tile_y * tile_size.h);
                    let end = Vec2u::new(
                        (start.x + tile_size.w).min(self.res.w),
                        (start.y + tile_size.h).min(self.res.h),
                    );
                    let tile_bounds = Aabru {
                        min: start,
                        max: end,
                    };

                    let tile = Tile::new(
                        self.progressive_epoch,
                        channels.iter().map(|c| c.kind()),
                        self.res,
                        tile_bounds,
                        filter,
                    );
                    tiles.push(tile);
                }
            }
        }

        self.integrate_tiles(tiles, |tile| {
            let mut rng = rand::thread_rng();
            let uniform = Uniform::new(-0.5, 0.5);

            let arena = DynamicArena::<'_, NonSend>::new_bounded();

            for x in tile.sample_bounds.min.x..tile.sample_bounds.max.x {
                for y in tile.sample_bounds.min.y..tile.sample_bounds.max.y {
                    for _ in 0..samples {
                        // let raster_pixel = Vec2u::new(x, y);
                        // sampler.begin_pixel(raster_pixel);

                        let uv_samp = Vec2::new(uniform.sample(&mut rng), uniform.sample(&mut rng));
                        let screen_coord = Vec2::new(x as f32 + 0.5, y as f32 + 0.5) + uv_samp;

                        let ndc = tile.screen_to_ndc_size * screen_coord;

                        let time = rng.gen_range(time_range.start, time_range.end);

                        let ray = camera.get_ray(ndc, time, &mut rng);
                        let (spect, first_intersection) =
                            integrator.integrate::<S>(world, ray, time, &mut rng, &arena);

                        tile.add_sample(screen_coord, spect.into(), first_intersection);
                    }
                }
            }
        });
    }

    fn integrate_tiles<F, FN>(&mut self, tiles: Vec<Tile<N, F>>, integrate_tile: FN)
    where
        F: Filter,
        FN: FnOnce(&mut Tile<N, F>) + Send + Sync + Copy,
    {
        let num_tiles = tiles.len();

        {
            let this = &*self;
            rayon::scope_fifo(|scope| {
                for (idx, mut tile) in tiles.into_iter().enumerate() {
                    scope.spawn_fifo(move |_| {
                        integrate_tile(&mut tile);

                        this.tile_finished(tile, num_tiles, idx)
                    })
                }
            });
        }

        while !self.this_epoch_tiles_finished.load(Ordering::Relaxed) == num_tiles {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        self.this_epoch_tiles_finished.store(0, Ordering::Relaxed);
        self.progressive_epoch += 1;
    }

    fn tile_finished<F: Filter>(&self, tile: Tile<N, F>, num_tiles: usize, tile_idx: usize) {
        if self.progressive_epoch != tile.epoch {
            panic!(
                "Epoch mismatch! Expected: {}, got: {}",
                self.progressive_epoch, tile.epoch
            );
        }

        let tile_percent = 1.0 / num_tiles as f32 * 100.0;
        let tile_percent_target = 5.0;
        let tile_divisor = (tile_percent_target / tile_percent).round() as usize;

        if tile_idx % tile_divisor == 0 {
            println!(
                "{}% finished...",
                (tile_idx as f32 * tile_percent).round() as u32
            );
        }

        let mut channels = self.channels.lock().unwrap();

        let Tile {
            channels: tile_channels,
            raster_bounds: tile_bounds,
            ..
        } = tile;

        for (tile_channel, channel) in tile_channels.iter().zip(channels.iter_mut()) {
            // Safe because we guarantee that we won't start modifying this chunk again
            // until the next epoch.
            channel
                .copy_from_tile(tile_channel, self.res, tile_bounds)
                .unwrap();
        }
    }
}