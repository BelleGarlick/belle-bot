use crate::types;

use rp2040_hal::gpio::AnyPin;
use rp2040_hal::pwm::SliceId;
use types as internal_types;

use embedded_hal::pwm::SetDutyCycle;

use rp2040_hal::gpio;
use rp2040_hal::pwm;
use rp2040_hal::pwm::Slice;

#[derive(PartialEq, Clone)]
pub enum Direction {
    Forward,
    Backward,
    Stopped,
    Left,
    Right,
}

pub struct Speed {
    direction: Direction,
    rate: u16,
}
impl Speed {
    pub fn get_rate(&self) -> u16 {
        self.rate
    }
    pub fn get_direction(&self) -> Direction {
        self.direction.clone()
    }
    fn forward(speed: u16) -> Speed {
        Speed {
            direction: Direction::Forward,
            rate: speed,
        }
    }
    fn backward(speed: u16) -> Speed {
        Speed {
            direction: Direction::Backward,
            rate: speed,
        }
    }
    fn stop() -> Speed {
        Speed {
            direction: Direction::Stopped,
            rate: 0,
        }
    }
    pub fn from_buff(buff: u8) -> Speed {
        if (97..107).contains(&buff) {
            Speed::forward(2500 * (buff - 97) as u16)
        } else if (107..117).contains(&buff) {
            Speed::backward(2500 * (buff - 117) as u16)
        } else if (117..127).contains(&buff) {
            Speed {
                direction: Direction::Left,
                rate: 2500 * (buff - 127) as u16,
            }
        } else if (127..137).contains(&buff) {
            Speed {
                direction: Direction::Right,
                rate: 2500 * (buff - 127) as u16,
            }
        } else {
            Speed::stop()
        }
    }
}

#[allow(dead_code)]
pub struct Motor<S: pwm::SliceId, TP: AnyPin, FP: AnyPin> {
    speed: Speed,
    to_pin: types::PWMChannelAPin<TP>,
    from_pin: types::PWMChannelBPin<FP>,
    pwm_slice: Slice<S, pwm::InputHighRunning>,
}
impl<S: SliceId, FP: AnyPin, TP: AnyPin> Motor<S, FP, TP>
where
    S: pwm::SliceId,
{
    pub fn forward(&mut self, speed: u16)
    where
        S: pwm::SliceId,
    {
        self.pwm_slice.channel_a.set_duty_cycle(0);
        let channel: &mut types::PWMChannelB<S> = &mut self.pwm_slice.channel_b;
        for i in 1..speed {
            channel.set_duty_cycle(i);
        }
        self.speed = Speed::forward(speed);
    }
    pub fn backwards(&mut self, speed: u16)
    where
        S: pwm::SliceId,
    {
        self.pwm_slice.channel_b.set_duty_cycle(0);
        let channel: &mut types::PWMChannelA<S> = &mut self.pwm_slice.channel_a;
        for i in 1..speed {
            channel.set_duty_cycle(i);
        }
        self.speed = Speed::backward(speed);
    }
    pub fn stop(&mut self)
    where
        S: pwm::SliceId,
    {
        self.pwm_slice.channel_a.set_duty_cycle(0);
        self.pwm_slice.channel_b.set_duty_cycle(0);
        self.speed = Speed::stop();
    }
    // TODO: be able to find pin id from pin instead of parsing in additional arg
    pub fn from(to_pin: TP, from_pin: FP, mut slice: Slice<S, pwm::FreeRunning>) -> Motor<S, TP, FP>
    where
        S: pwm::SliceId,
        TP: gpio::AnyPin,
        FP: gpio::AnyPin,
        <FP as gpio::AnyPin>::Id: pwm::ValidPwmInputPin<S>,
        <TP as gpio::AnyPin>::Id: pwm::ValidPwmOutputPin<S, pwm::A>,
    {
        // We set the ph (phase) to the correct mode.
        // We enable the slice.
        // We set the slice to high running input meaning we read the input and output consistently from
        //  channel b to channel a
        slice.set_ph_correct();
        slice.enable();
        let mut slice: Slice<S, pwm::InputHighRunning> = slice.into_mode::<pwm::InputHighRunning>();

        // Slice is a generic slice type provided by the pwm slice
        // FreeRunning is the slice mode where the counter is continuous
        // B is the channel output type, where current flows out
        // set the input and output of the pwm signal, pin is consumed by this and the object is lost
        let channel_pin_b: internal_types::PWMChannelBPin<FP> =
            slice.channel_b.input_from(from_pin);
        let channel_pin_a: internal_types::PWMChannelAPin<TP> = slice.channel_a.output_to(to_pin);

        // return info on the motor
        Motor {
            speed: Speed::stop(),
            to_pin: channel_pin_a,
            from_pin: channel_pin_b,
            pwm_slice: slice,
        }
    }
}
