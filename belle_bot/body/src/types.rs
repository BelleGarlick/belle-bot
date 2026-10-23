use rp2040_hal::gpio::{
    bank0::{Gpio0, Gpio1},
    FunctionUart, Pin, PullDown,
};
use rp2040_hal::typelevel;
use rp2040_hal::uart;

// TODO: refactor so that this type can take any value uart GPIO pin.
pub type UartPins = uart::Pins<
    typelevel::OptionTSome<Pin<Gpio0, FunctionUart, PullDown>>,
    typelevel::OptionTSome<Pin<Gpio1, FunctionUart, PullDown>>,
    typelevel::OptionTNone,
    typelevel::OptionTNone,
>;
