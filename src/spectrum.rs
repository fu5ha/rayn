use crate::math::Vec3;
use image;

#[derive(Clone, Copy, Debug)]
pub struct Spectrum(pub Vec3);

impl Spectrum {
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Spectrum(Vec3::new(r, g, b))
    }

    pub fn zero() -> Self {
        Spectrum(Vec3::zero())
    }

    pub fn one() -> Self {
        Spectrum(Vec3::one())
    }

    pub fn is_black(&self) -> bool {
        self.max_channel() < 0.0001
    }

    pub fn max_channel(&self) -> f32 {
        self.0.reduce_partial_max()
    }

    pub fn gamma_correct(self, gamma: f32) -> Self {
        Spectrum(Vec3::new(
            self.0.x.powf(1.0 / gamma),
            self.0.y.powf(1.0 / gamma),
            self.0.z.powf(1.0 / gamma),
        ))
    }
}

impl From<Spectrum> for image::Rgb<u8> {
    fn from(col: Spectrum) -> Self {
        image::Rgb([
            (col.0.x * 255.0).min(255.0).max(0.0) as u8,
            (col.0.y * 255.0).min(255.0).max(0.0) as u8,
            (col.0.z * 255.0).min(255.0).max(0.0) as u8,
        ])
    }
}

impl ::std::ops::Add for Spectrum {
    type Output = Spectrum;

    fn add(self, other: Spectrum) -> Spectrum {
        Spectrum(self.0 + other.0)
    }
}

impl ::std::ops::Div<f32> for Spectrum {
    type Output = Spectrum;

    fn div(self, other: f32) -> Spectrum {
        Spectrum(self.0 / other)
    }
}

impl std::ops::DivAssign<f32> for Spectrum {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs
    }
}

impl ::std::ops::Mul<f32> for Spectrum {
    type Output = Spectrum;

    fn mul(self, other: f32) -> Spectrum {
        Spectrum(self.0 * other)
    }
}

impl std::ops::MulAssign<f32> for Spectrum {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs
    }
}

impl ::std::ops::Mul<Spectrum> for Spectrum {
    type Output = Spectrum;

    fn mul(self, other: Spectrum) -> Spectrum {
        Spectrum(self.0 * other.0)
    }
}

impl std::ops::MulAssign for Spectrum {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs
    }
}


impl std::ops::AddAssign for Spectrum {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}
