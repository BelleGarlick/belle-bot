use embedded_hal::pwm::SetDutyCycle;
use rp2040_hal;

use rp2040_hal::pwm;
use rp2040_hal::gpio;
use rp2040_hal::pwm::Slice;

pub enum Direction {
    FORWARD,
    BACKWARD,
    STOPPED
} impl Clone for Direction {
    fn clone(&self) -> Direction {
        match self {
            Direction::FORWARD => Direction::FORWARD,
            Direction::BACKWARD => Direction::BACKWARD,
            Direction::STOPPED => Direction::STOPPED,
        }
    }
}

#[allow(dead_code)]
pub struct Motor<S: pwm::SliceId> {
    state: Direction,
    pin: u8,
    pwm_slice: Slice<S, pwm::FreeRunning>
}
impl <'a, S> Motor<S>
    where 
        S: pwm::SliceId  {
    pub fn forward(&mut self)
    where 
        S: pwm::SliceId  
    {
        self.pwm_slice.channel_a.set_duty_cycle(0);
        let channel: &mut pwm::Channel<Slice<S, pwm::FreeRunning>, pwm::B> = &mut self.pwm_slice.channel_b;
        for i in 1..25000 {
            channel.set_duty_cycle(i);
        }
        self.state = Direction::FORWARD;
    }
    pub fn backwards(&mut self)
    where 
        S: pwm::SliceId 
    {
        self.pwm_slice.channel_b.set_duty_cycle(0);
        let channel: &mut pwm::Channel<Slice<S, pwm::FreeRunning>, pwm::A> = &mut self.pwm_slice.channel_a;
        for i in (1..25000).rev() {
            channel.set_duty_cycle(i);
        }
        self.state = Direction::BACKWARD;
    }
    pub fn stop(&mut self)
    where 
        S: pwm::SliceId 
    {
        self.pwm_slice.channel_a.set_duty_cycle(0);
        self.pwm_slice.channel_b.set_duty_cycle(0);
        self.state = Direction::STOPPED;
    }
    pub fn from<P>(pin: P, pin_num: u8, mut pwm_slice: Slice<S, pwm::FreeRunning>) -> Motor<S>
    where 
        S: pwm::SliceId,
        P: gpio::AnyPin,
        <P as gpio::AnyPin>::Id: pwm::ValidPwmOutputPin<S, pwm::B>
    {
        // creates a central pulsing wave for PWM ph is phase correct which means 
        // the wave form timer counts all the way up and then all the way back down again giving a nice curve
        pwm_slice.set_ph_correct();
        pwm_slice.enable();

        // Slice is a generic slice type provided by the pwm slice
        // FreeRunning is the slice mode where the counter is continuous
        // B is the channel output type, where current flows out
        let channel: &mut pwm::Channel<Slice<S, pwm::FreeRunning>, pwm::B>  = &mut pwm_slice.channel_b;

        // set the output of the pwm signal, pin is consumed by this and the object is lost
        channel.output_to(pin);
        
        // return info on the motor
        Motor {
            state: Direction::STOPPED,
            pin: pin_num,
            pwm_slice: pwm_slice
        }
    }
}
