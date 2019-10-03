use vek::Clamp;

use crate::math::Vec3;
use std::fmt::Debug;
use std::iter::*;
use std::ops::*;

pub trait IsSpectrum:
    Add<Self, Output = Self>
    + AddAssign<Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Mul<f32, Output = Self>
    + Div<f32, Output = Self>
    + DivAssign<f32>
    + Sum
    + PartialEq
    + Sized
    + Clone
    + Copy
    + Debug
    + Into<Xyz>
    + From<Xyz>
    + Send
    + Sync
{
    fn zero() -> Self;
    fn one() -> Self;
    fn is_black(&self) -> bool;
    fn is_nan(&self) -> bool;
    fn max_channel(&self) -> f32;
}

type VekRgb = vek::vec::Rgb<f32>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgb(pub VekRgb);

impl From<Vec3> for Rgb {
    fn from(v: Vec3) -> Self {
        Rgb(VekRgb::from(v))
    }
}

impl Rgb {
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Rgb(VekRgb::new(r, g, b))
    }

    #[allow(dead_code)]
    pub fn gamma_corrected(&self, gamma: f32) -> Self {
        Rgb(self.0.map(|x| x.powf(1.0 / gamma)))
    }

    #[allow(dead_code)]
    pub fn saturated(&self) -> Rgb {
        Rgb(self.0.map(|x| Clamp::clamped01(x)))
    }
}

impl Deref for Rgb {
    type Target = VekRgb;
    fn deref(&self) -> &VekRgb {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Xyz(Vec3);

impl Deref for Xyz {
    type Target = Vec3;
    fn deref(&self) -> &Vec3 {
        &self.0
    }
}

impl Xyz {
    #[allow(dead_code)]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Xyz(Vec3::new(x, y, z))
    }

    #[allow(dead_code)]
    pub fn gamma_corrected(&self, gamma: f32) -> Self {
        Xyz(self.0.map(|x| x.powf(1.0 / gamma)))
    }

    #[allow(dead_code)]
    pub fn saturated(&self) -> Self {
        Xyz(self.0.map(|x| Clamp::clamped01(x)))
    }
}

impl From<Rgb> for Xyz {
    fn from(rgb: Rgb) -> Self {
        Xyz(Vec3 {
            // x: 0.412453 * rgb.r + 0.357580 * rgb.g + 0.180423 * rgb.b,
            // y: 0.212671 * rgb.r + 0.715160 * rgb.g + 0.072169 * rgb.b,
            // z: 0.019334 * rgb.r + 0.119193 * rgb.g + 0.950227 * rgb.b,
            x: 0.4360747 * rgb.r + 0.3850649 * rgb.g + 0.1430804 * rgb.b,
            y: 0.2225045 * rgb.r + 0.7168786 * rgb.g + 0.0606169 * rgb.b,
            z: 0.0139322 * rgb.r + 0.0971045 * rgb.g + 0.7141733 * rgb.b,
        })
    }
}

impl From<Xyz> for Rgb {
    fn from(xyz: Xyz) -> Self {
        Rgb::new(
            3.1338561 * xyz.x - 1.6168667 * xyz.y - 0.4906146 * xyz.z,
            -0.9787684 * xyz.x + 1.9161415 * xyz.y + 0.0334540 * xyz.z,
            0.0719453 * xyz.x - 0.2289914 * xyz.y + 1.4052427 * xyz.z,
        )
    }
}

impl IsSpectrum for Xyz {
    fn zero() -> Self {
        Xyz(Vec3::zero())
    }

    fn one() -> Self {
        Xyz(Vec3::one())
    }

    fn is_black(&self) -> bool {
        self.max_channel() < 0.0001
    }

    fn is_nan(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }

    fn max_channel(&self) -> f32 {
        self.0.reduce_partial_max()
    }
}

impl Sum for Xyz {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Xyz::zero(), |a, b| a + b)
    }
}

macro_rules! impl_wrapper_ops {
    ($wrapper_t:ident) => {
        impl ::std::ops::Add for $wrapper_t {
            type Output = $wrapper_t;

            fn add(self, other: $wrapper_t) -> $wrapper_t {
                $wrapper_t(self.0 + other.0)
            }
        }

        impl std::ops::AddAssign for $wrapper_t {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs
            }
        }

        impl ::std::ops::Sub for $wrapper_t {
            type Output = $wrapper_t;

            fn sub(self, other: $wrapper_t) -> $wrapper_t {
                $wrapper_t(self.0 - other.0)
            }
        }

        impl std::ops::SubAssign for $wrapper_t {
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs
            }
        }

        impl ::std::ops::Div<f32> for $wrapper_t {
            type Output = $wrapper_t;

            fn div(self, other: f32) -> $wrapper_t {
                $wrapper_t(self.0 / other)
            }
        }

        impl std::ops::DivAssign<f32> for $wrapper_t {
            fn div_assign(&mut self, rhs: f32) {
                *self = *self / rhs
            }
        }

        impl ::std::ops::Mul<f32> for $wrapper_t {
            type Output = $wrapper_t;

            fn mul(self, other: f32) -> $wrapper_t {
                $wrapper_t(self.0 * other)
            }
        }

        impl std::ops::MulAssign<f32> for $wrapper_t {
            fn mul_assign(&mut self, rhs: f32) {
                *self = *self * rhs
            }
        }

        impl ::std::ops::Mul<$wrapper_t> for $wrapper_t {
            type Output = $wrapper_t;

            fn mul(self, other: $wrapper_t) -> $wrapper_t {
                $wrapper_t(self.0 * other.0)
            }
        }

        impl std::ops::MulAssign for $wrapper_t {
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs
            }
        }
    };
}

impl_wrapper_ops!(Xyz);
impl_wrapper_ops!(Rgb);
