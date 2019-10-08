pub trait Sampler {
    fn begin_pixel(&mut self, pixel: Vec2u);
    fn request_samples_2d(&mut self, samples: usize);
    fn request_samples_1d(&mut self, samples: usize);
}
