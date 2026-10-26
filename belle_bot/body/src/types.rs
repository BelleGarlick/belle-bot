use rp2040_hal::gpio::{
    bank0::{Gpio0, Gpio1},
    AnyPin, FunctionPwm, FunctionUart, Pin, PullDown,
};
use rp2040_hal::pwm::{Channel, InputHighRunning, Slice, A, B};
use rp2040_hal::typelevel;
use rp2040_hal::uart;

// TODO: refactor so that this type can take any value uart GPIO pin.
pub type UartPins = uart::Pins<
    typelevel::OptionTSome<Pin<Gpio0, FunctionUart, PullDown>>,
    typelevel::OptionTSome<Pin<Gpio1, FunctionUart, PullDown>>,
    typelevel::OptionTNone,
    typelevel::OptionTNone,
>;

pub type PWMChannelA<S> = Channel<Slice<S, InputHighRunning>, A>;

pub type PWMChannelB<S> = Channel<Slice<S, InputHighRunning>, B>;

pub type PWMChannelAPin<TP> = Pin<<TP as AnyPin>::Id, FunctionPwm, <TP as AnyPin>::Pull>;

pub type PWMChannelBPin<FP> = Pin<<FP as AnyPin>::Id, FunctionPwm, <FP as AnyPin>::Pull>;
