use crate::math::Vec3;
use image;
use std::ops::*;
use std::fmt::Debug;

pub trait IsSpectrum: 
    Add<Self, Output=Self> + AddAssign<Self> + Sub<Self, Output=Self> + SubAssign<Self>
    + Mul<Self, Output=Self> + MulAssign<Self>
    + Mul<f32, Output=Self> + Div<f32, Output=Self>
    + PartialEq + Sized + Clone + Copy + Debug + Into<image::Rgb<u8>> + Send + Sync
{
    fn zero() -> Self;
    fn one() -> Self;
    fn is_black(&self) -> bool;
    fn max_channel(&self) -> f32;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RGBSpectrum(pub Vec3);

impl RGBSpectrum {
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        RGBSpectrum(Vec3::new(r, g, b))
    }

    pub fn gamma_correct(self, gamma: f32) -> Self {
        RGBSpectrum(Vec3::new(
            self.0.x.powf(1.0 / gamma),
            self.0.y.powf(1.0 / gamma),
            self.0.z.powf(1.0 / gamma),
        ))
    }
}

impl IsSpectrum for RGBSpectrum {
    fn zero() -> Self {
        RGBSpectrum(Vec3::zero())
    }

    fn one() -> Self {
        RGBSpectrum(Vec3::one())
    }

    fn is_black(&self) -> bool {
        self.max_channel() < 0.0001
    }

    fn max_channel(&self) -> f32 {
        self.0.reduce_partial_max()
    }
}

impl From<RGBSpectrum> for image::Rgb<u8> {
    fn from(col: RGBSpectrum) -> Self {
        image::Rgb([
            (col.0.x * 255.0).min(255.0).max(0.0) as u8,
            (col.0.y * 255.0).min(255.0).max(0.0) as u8,
            (col.0.z * 255.0).min(255.0).max(0.0) as u8,
        ])
    }
}

impl ::std::ops::Add for RGBSpectrum {
    type Output = RGBSpectrum;

    fn add(self, other: RGBSpectrum) -> RGBSpectrum {
        RGBSpectrum(self.0 + other.0)
    }
}

impl std::ops::AddAssign for RGBSpectrum {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl ::std::ops::Sub for RGBSpectrum {
    type Output = RGBSpectrum;

    fn sub(self, other: RGBSpectrum) -> RGBSpectrum {
        RGBSpectrum(self.0 - other.0)
    }
}

impl std::ops::SubAssign for RGBSpectrum {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs
    }
}

impl ::std::ops::Div<f32> for RGBSpectrum {
    type Output = RGBSpectrum;

    fn div(self, other: f32) -> RGBSpectrum {
        RGBSpectrum(self.0 / other)
    }
}

impl std::ops::DivAssign<f32> for RGBSpectrum {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs
    }
}

impl ::std::ops::Mul<f32> for RGBSpectrum {
    type Output = RGBSpectrum;

    fn mul(self, other: f32) -> RGBSpectrum {
        RGBSpectrum(self.0 * other)
    }
}

impl std::ops::MulAssign<f32> for RGBSpectrum {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs
    }
}

impl ::std::ops::Mul<RGBSpectrum> for RGBSpectrum {
    type Output = RGBSpectrum;

    fn mul(self, other: RGBSpectrum) -> RGBSpectrum {
        RGBSpectrum(self.0 * other.0)
    }
}

impl std::ops::MulAssign for RGBSpectrum {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs
    }
}

