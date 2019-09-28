use generic_array::{ GenericArray, ArrayLength };
use dynamic_arena::{ DynamicArena, NonSend };

use rand::distributions::Uniform;
use rand::prelude::*;

use crate::camera::CameraHandle;
use crate::math::{ Vec2u, Vec2, Vec3, Aabr, Extent2u, Aabru };
use crate::spectrum::{ Xyz, Rgb, IsSpectrum };
use crate::integrator::{ Integrator };
use crate::world::World;

use std::sync::{ atomic::{ AtomicUsize, Ordering }, Mutex };
use std::ops::Range;
use std::cell::UnsafeCell;
use std::collections::HashMap;

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

        pub enum ChannelStorage {
            $($name(Vec<$storage>),)+
        }

        impl ChannelStorage {
            pub fn kind(&self) -> ChannelKind {
                match *self {
                    $( ChannelStorage::$name(_) => ChannelKind::$name, )+
                }
            }

            pub fn copy_from_tile(&mut self, other: &ChannelStorage, full_res: Extent2u, tile_bounds: Aabru) -> Result<(), ()> {
                let extent = tile_bounds.size();
                match (self, other) {
                    $( (ChannelStorage::$name(this_buf), ChannelStorage::$name(tile_buf)) => {
                        for x in 0..extent.w {
                            for y in 0..extent.h {
                                let tile_idx = x + y * extent.w;
                                let this_idx = (tile_bounds.min.x + x) + (tile_bounds.min.y + y) * full_res.w;
                                this_buf[this_idx] = tile_buf[tile_idx];
                            }
                        }
                        Ok(())
                    }, )+
                    _ => Err(())
                }
            }
        }

        pub enum ChannelRefMut<'a> {
            $($name(&'a mut [$storage]),)+
        }

        impl<'a> ChannelRefMut<'a> {
            pub fn from_storage_mut(storage: &'a mut ChannelStorage) -> Self {
                match storage {
                    $( ChannelStorage::$name(buf) => ChannelRefMut::$name(buf.as_mut()), )+
                }
            }
        }

        trait ChannelInit {
            fn init() -> Self;
        }

        impl ChannelStorage {
            fn new(kind: ChannelKind, res: Extent2u) -> Self {
                use ChannelKind::*;
                match kind {
                    $($name => Self::$name(vec![$initialize; res.w * res.h]),)+
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
    }
}

pub type UnsafeChannelStorage = UnsafeCell<ChannelStorage>;

unsafe impl<N: ArrayLength<UnsafeChannelStorage>> Send for Tile<N> {}
unsafe impl<N: ArrayLength<UnsafeChannelStorage>> Sync for Tile<N> {}

pub struct Tile<N: ArrayLength<UnsafeChannelStorage>> {
    epoch: AtomicUsize,
    channels: GenericArray<UnsafeChannelStorage, N>,
    pixel_bounds: Aabru,
    uv_bounds: Aabr,
}

pub struct Film<N: ArrayLength<ChannelStorage> + ArrayLength<UnsafeChannelStorage>> {
    channel_indices: HashMap<ChannelKind, usize>,
    channels: Mutex<GenericArray<ChannelStorage, N>>,
    tiles: Vec<Tile<N>>,
    progressive_epoch: usize,
    this_epoch_tiles_finished: AtomicUsize,
    res: Extent2u,
}

impl<'a, N: ArrayLength<ChannelStorage> + ArrayLength<UnsafeChannelStorage>> Film <N> { 
    pub fn new(
        channels: &[ChannelKind],
        res: Extent2u,
    ) -> Result<Self, String> {
        let mut channel_indices = HashMap::new();
        for (i, kind) in channels.iter().enumerate() {
            if let Some(_) = channel_indices.insert(*kind, i) {
                return Err(String::from(format!("Attempted to create multiple {:?} channels", *kind)));
            }
        }
        Ok(Film {
            channel_indices,
            channels: Mutex::new(GenericArray::from_exact_iter(channels.into_iter().map(|kind| {
                ChannelStorage::new(*kind, res)
            })).expect("Generic type length does not match the number of channels.")),
            tiles: Vec::new(),
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
                            let mut img = image::RgbaImage::new(self.res.w as u32, self.res.h as u32);
                            for (x, y, pixel) in img.enumerate_pixels_mut() {
                                let idx = x as usize + (self.res.h - 1 - y as usize) * self.res.w;
                                let col = color_buf[idx];
                                let a = alpha_buf[idx];
                                let rgb = Rgb::from(col).gamma_corrected(2.2) * a;
                                *pixel = image::Rgba([
                                    (rgb.r * 255.0).min(255.0).max(0.0) as u8,
                                    (rgb.g * 255.0).min(255.0).max(0.0) as u8,
                                    (rgb.b * 255.0).min(255.0).max(0.0) as u8,
                                    (a * 255.0).min(255.0).max(0.0) as u8,
                                ]);
                            }
                            let filename = output_folder.as_ref().join(format!("{}_color.png", base_name.clone()));
                            println!("Saving to {}...", filename.display());
                            img.save(filename).unwrap();
                        },
                        (Some(&color_idx), _, Some(&bg_idx), false) => {
                            let color_buf = channel_storage_index!(channels, Color, color_idx);
                            let bg_buf = channel_storage_index!(channels, Background, bg_idx);
                            let mut img = image::RgbImage::new(self.res.w as u32, self.res.h as u32);
                            for (x, y, pixel) in img.enumerate_pixels_mut() {
                                let i = x as usize + (self.res.h - 1 - y as usize) * self.res.w;
                                let col = color_buf[i];
                                let bg = bg_buf[i];
                                let rgb = Rgb::from(col + bg).gamma_corrected(2.2);
                                *pixel = image::Rgb([
                                    (rgb.r * 255.0).min(255.0).max(0.0) as u8,
                                    (rgb.g * 255.0).min(255.0).max(0.0) as u8,
                                    (rgb.b * 255.0).min(255.0).max(0.0) as u8,
                                ]);
                            }
                            let filename = output_folder.as_ref().join(format!("{}_color.png", base_name.clone()));
                            println!("Saving to {}...", filename.display());
                            img.save(filename).unwrap();
                        },
                        (Some(&color_idx), _, None, false) => {
                            let color_buf = channel_storage_index!(channels, Color, color_idx);
                            let mut img = image::RgbImage::new(self.res.w as u32, self.res.h as u32);
                            for (x, y, pixel) in img.enumerate_pixels_mut() {
                                let idx = x as usize + (self.res.h - 1 - y as usize) * self.res.w;
                                let rgb = Rgb::from(color_buf[idx]).gamma_corrected(2.2);
                                *pixel = image::Rgb([
                                    (rgb.r * 255.0).min(255.0).max(0.0) as u8,
                                    (rgb.g * 255.0).min(255.0).max(0.0) as u8,
                                    (rgb.b * 255.0).min(255.0).max(0.0) as u8,
                                ]);
                            }
                            let filename = output_folder.as_ref().join(format!("{}_color.png", base_name.clone()));
                            println!("Saving to {}...", filename.display());
                            img.save(filename).unwrap();
                        },
                        _ => return Err(String::from("Attempted to write Color channel with insufficient channels")),
                    }
                },
                ChannelKind::Background => {
                    let idx = *self.channel_indices.get(&ChannelKind::Background)
                        .ok_or(String::from("Attempted to write Background channel but it didn't exist"))?;
                    let buf = channel_storage_index!(channels, Background, idx);
                    let mut img = image::RgbImage::new(self.res.w as u32, self.res.h as u32);
                    for (x, y, pixel) in img.enumerate_pixels_mut() {
                        let idx = x as usize + (self.res.h - 1 - y as usize) * self.res.w;
                        let rgb = Rgb::from(buf[idx]).gamma_corrected(2.2);
                        *pixel = image::Rgb([
                            (rgb.r * 255.0).min(255.0).max(0.0) as u8,
                            (rgb.g * 255.0).min(255.0).max(0.0) as u8,
                            (rgb.b * 255.0).min(255.0).max(0.0) as u8,
                        ]);
                    }
                    let filename = output_folder.as_ref().join(format!("{}_background.png", base_name.clone()));
                    println!("Saving to {}...", filename.display());
                    img.save(filename).unwrap();
                },
                ChannelKind::WorldNormal => {
                    let idx = *self.channel_indices.get(&ChannelKind::WorldNormal)
                        .ok_or(String::from("Attempted to write WorldNormal channel but it didn't exist"))?;
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
                    let filename = output_folder.as_ref().join(format!("{}_normal.png", base_name.clone()));
                    println!("Saving to {}...", filename.display());
                    img.save(filename).unwrap();
                },
                ChannelKind::Alpha => {
                    let idx = *self.channel_indices.get(&ChannelKind::Alpha)
                        .ok_or(String::from("Attempted to write Alpha channel but it didn't exist"))?;
                    let buf = channel_storage_index!(channels, Alpha, idx);
                    let mut img = image::GrayImage::new(self.res.w as u32, self.res.h as u32);
                    for (x, y, pixel) in img.enumerate_pixels_mut() {
                        let idx = x as usize + (self.res.h - 1 - y as usize) * self.res.w;
                        let a = buf[idx];
                        *pixel = image::Luma([
                            (a * 255.0).min(255.0).max(0.0) as u8,
                        ]);
                    }
                    let filename = output_folder.as_ref().join(format!("{}_alpha.png", base_name.clone()));
                    println!("Saving to {}...", filename.display());
                    img.save(filename).unwrap();
                },
            }
        }
        Ok(())
    }
}

impl<'a, N: ArrayLength<ChannelStorage> + ArrayLength<UnsafeChannelStorage> + ArrayLength<ChannelRefMut<'a>>> Film<N>  {
    pub fn render_frame_into<I: Integrator, S: IsSpectrum>(
        &'a mut self,
        world: &World<S>,
        camera: CameraHandle,
        integrator: I,
        tile_size: Extent2u,
        time_range: Range<f32>,
        samples: usize,
    ) {
        let camera = world.cameras.get(camera);
        self.tiles.clear();

        let rem = Vec2u::new(
            (self.res.w) % tile_size.w,
            (self.res.h) % tile_size.h);
        {
            let channels = self.channels.lock().unwrap();
            for tile_x in 0..((self.res.w + rem.x) / tile_size.w) {
                for tile_y in 0..((self.res.h + rem.y) / tile_size.h) {
                    let start = Vec2u::new(tile_x * tile_size.w, tile_y * tile_size.h);
                    let end = Vec2u::new(
                        (start.x + tile_size.w).min(self.res.w),
                        (start.y + tile_size.h).min(self.res.h));
                    let tile_bounds = Aabru {
                        min: start,
                        max: end,
                    };
                    let actual_tile_size = tile_bounds.size();

                    let uv_start = Vec2::new(start.x as f32 / self.res.w as f32, start.y as f32 / self.res.h as f32);
                    let uv_end = Vec2::new(end.x as f32 / self.res.w as f32, end.y as f32 / self.res.h as f32);
                    let uv_bounds = Aabr {
                        min: uv_start,
                        max: uv_end
                    };

                    let tile_channels = GenericArray::from_exact_iter(channels.iter().map(|channel| {
                        UnsafeCell::new(ChannelStorage::new(channel.kind(), actual_tile_size))
                    })).unwrap();

                    self.tiles.push(Tile {
                        epoch: AtomicUsize::new(0),
                        channels: tile_channels,
                        pixel_bounds: tile_bounds,
                        uv_bounds,
                    });
                }
            }
        }

        self.integrate_tiles(|mut tile, tile_extent, uv_bounds| {
            let mut rng = rand::thread_rng();
            let uniform = Uniform::new(0.0, 1.0);

            let arena = DynamicArena::<'_, NonSend>::new_bounded();

            let tile_extent_f32 = Vec2::new(tile_extent.w as f32, tile_extent.h as f32);

            for x in 0..tile_extent.w {
                for y in 0..tile_extent.h {
                    let mut col_spect = S::zero();
                    let mut a = 0.0;
                    let mut back_spect = S::zero();
                    let mut normals = Vec3::zero();
                    for _ in 0..samples {
                        let r = Vec2::new(uniform.sample(&mut rng), uniform.sample(&mut rng));

                        let tile_xy = Vec2::new(x as f32, y as f32) + r;
                        let uv_offset = tile_xy / tile_extent_f32 * uv_bounds.size();
                        let uv = uv_bounds.min + uv_offset;

                        let time = rng.gen_range(time_range.start, time_range.end);

                        let ray = camera.get_ray(uv, time, &mut rng);
                        integrator.integrate::<S>(
                            world,
                            ray,
                            time,
                            &mut col_spect,
                            &mut a,
                            &mut back_spect,
                            &mut normals,
                            &mut rng,
                            &arena);
                    }

                    col_spect = col_spect / samples as f32;
                    a = a / samples as f32;
                    back_spect = back_spect / samples as f32;
                    normals /= samples as f32;

                    let idx = x + y * tile_extent.w;
                    for channel_ref in tile.iter_mut() {
                        match channel_ref {
                            ChannelRefMut::Color(buf) => {
                                buf[idx] = col_spect.into();
                            },
                            ChannelRefMut::Alpha(buf) => {
                                buf[idx] = a;
                            },
                            ChannelRefMut::Background(buf) => {
                                buf[idx] = back_spect.into();
                            },
                            ChannelRefMut::WorldNormal(buf) => {
                                buf[idx] = normals;
                            }
                        }
                    }
                }
            }
        });
    }

    fn integrate_tiles<F>(&'a mut self, integrate_tile: F)
        where F: FnOnce(GenericArray<ChannelRefMut<'a>, N>, Extent2u, Aabr) + Send + Sync + Copy
    {
        let tile_infos = self
            .tiles
            .iter()
            .enumerate()
            .map(|(idx, Tile { channels, pixel_bounds, uv_bounds, .. }) | {
                let tile_channels = GenericArray::from_exact_iter(channels.iter().map(|channel| {
                    // Safe because we guarantee that nobody else is accessing this specific
                    // slice at the same time, and we do not modify or read the underlying Vec
                    // until after this ref goes out of scope.
                    ChannelRefMut::from_storage_mut(unsafe { &mut *channel.get() })
                })).unwrap();

                (idx, tile_channels, pixel_bounds.size(), *uv_bounds)
            }).collect::<Vec<_>>();
        
        let num_tiles = self.tiles.len();

        {
            let epoch = self.progressive_epoch;
            let this = &*self;
            rayon::scope_fifo(|scope| {
                for (tile_idx, tile_channels, pixel_size, uv_bounds) in tile_infos.into_iter() {
                    scope.spawn_fifo(move |_| {
                        integrate_tile(tile_channels, pixel_size, uv_bounds);

                        this.tile_finished(tile_idx, epoch)
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

    fn tile_finished(&self, tile_idx: usize, epoch: usize) {
        let tile = &self.tiles[tile_idx];
        let tile_epoch = tile.epoch.fetch_add(1, Ordering::Relaxed);
        if epoch != tile_epoch {
            panic!("Epoch mismatch! Expected: {}, got: {}", tile_epoch, epoch)
        }

        let tile_percent = 1.0 / self.tiles.len() as f32 * 100.0;
        let tile_percent_target = 5.0;
        let tile_divisor = (tile_percent_target / tile_percent).round() as usize;
        
        if tile_idx % tile_divisor == 0 {
            println!(
                "{}% finished...",
                (tile_idx as f32 * tile_percent).round() as u32
            );
        }

        let mut channels = self.channels.lock().unwrap();

        let Tile { channels: tile_channels, pixel_bounds: tile_bounds, .. } = tile;

        for (tile_channel, channel) in tile_channels.iter().zip(channels.iter_mut()) {
            // Safe because we guarantee that we won't start modifying this chunk again
            // until the next epoch.
            let tile_channel = unsafe { &*tile_channel.get() };
            channel.copy_from_tile(tile_channel, self.res, *tile_bounds).unwrap();
        }
    }
}