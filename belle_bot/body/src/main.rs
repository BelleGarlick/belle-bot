#![no_std]
#![no_main]

/// https://github.com/khoih-prog/RP2040_PWM/blob/main/README.md <- PWM output
/// CONNECTIONS:
///     Battery:
///         12v DC male plug (ciggerette lighter) goes into 12v battery port
///     
///     MDD10A:
///         Power port B- is connected to Negative Anode of cigarette lighter (Black wire)
///         Power port B+ is connected to Positive Cathode of cigarette lighter (Red wire)
///
///         Port M2A is connected to the Cathode of a 12v brush motor
///         Port M2B is connected to the Anode of a 12v brush motor
///         
///         GND is connected to GPIO22 on pico
///         PWM2 is connected to GPIO21 on pico
///         DIR2 is connected to GPIO20 on pico
///         
///         
///     PICO H (rp2040):
///         pico is connected via micro usb to a power 3.3-5v power source when in production
///         pico is connected via micro usb to a power a laptop when in dev, draw from laptop is sufficient
///
///     
///     Drawing:
///         Bat     DVR/Motor    Motor          DVR/CTRL      CTRL          JETSON
///                            _______
///                        ->[-]      \___
///                        | [+]______/
///                [M1B] --]  |
///         [+]    [M1A] ---->|                 [DRV1]   ->   GPIO18
///          |       |                          [PWM1]   ->   GPIO19
///          | ---> [+]                         [DRV2]   ->   GPIO20
///          | ---> [-]<--------------------    [PWM2]   ->   GPIO21
///          |       |                     |     [GRD]   ->   GND
///         [-]    [M2A] ---->|            |                  GPIO0[TX]  ->  RX
///                [M1B] --]  | _____      |             ->   GPIO1[RX]  ->  TX
///                        | [+]      \___ |             ->   VSYS       ->  3.3/5v OUT
///                        ->[-]______/    |                  GND        ->  GND
///                                        ----------------   GND
///
mod motor;

use defmt_rtt as _;
use panic_probe as _;

use embedded_hal::digital::OutputPin;
use rp2040_hal::pac;
use rp2040_hal::pwm;
use rp2040_hal::uart;
use rp2040_hal::{gpio::{Pins, FunctionUart}, pwm::Slices, Sio, Watchdog};
use rp2040_hal::fugit::RateExtU32;
use rp2040_hal::Clock;


#[link_section = ".boot2"]
#[used]
pub static BOOT2: [u8; 256] = rp2040_boot2::BOOT_LOADER_W25Q080;

#[cortex_m_rt::entry]
fn main() -> ! {
    // take the peripherals for the device
    let mut pac: pac::Peripherals = pac::Peripherals::take().unwrap();
    // prevents time out
    let mut watchdog: Watchdog = Watchdog::new(pac.WATCHDOG);
    // inialise a clock to keep track of timings
    let clocks: rp2040_hal::clocks::ClocksManager = rp2040_hal::clocks::init_clocks_and_plls(
        12_000_000u32,
        pac.XOSC,
        pac.CLOCKS,
        pac.PLL_SYS,
        pac.PLL_USB,
        &mut pac.RESETS,
        &mut watchdog,
    )
    .unwrap();

    let sio: Sio = Sio::new(pac.SIO);
    // initalise the pins
    let pins: Pins = Pins::new(
        pac.IO_BANK0,
        pac.PADS_BANK0,
        sio.gpio_bank0,
        &mut pac.RESETS,
    );

    let pwm_slices: Slices = Slices::new(pac.PWM, &mut pac.RESETS);

    // sets up two
    let mut motor_one: motor::Motor<pwm::Pwm1> = motor::Motor::from(pins.gpio19, 19, pwm_slices.pwm1);
    let mut motor_two: motor::Motor<pwm::Pwm2> = motor::Motor::from(pins.gpio21, 21, pwm_slices.pwm2);

    // sets gpio25 onboard led to high so that we can see the pico has been flashed.
    let mut led_pin: rp2040_hal::gpio::Pin<
        rp2040_hal::gpio::bank0::Gpio25,
        rp2040_hal::gpio::FunctionSio<rp2040_hal::gpio::SioOutput>,
        rp2040_hal::gpio::PullDown,
    > = pins.gpio25.into_push_pull_output();

    let uart_pins = (
        pins.gpio0.into_function(),
        pins.gpio1.into_function(),
    );

    // setting up uart connections
    // we connect the output
    let uart: uart::UartPeripheral<uart::Enabled, pac::UART0,
        (
            rp2040_hal::gpio::Pin<rp2040_hal::gpio::bank0::Gpio0, FunctionUart, rp2040_hal::gpio::PullDown>,
            rp2040_hal::gpio::Pin<rp2040_hal::gpio::bank0::Gpio1, FunctionUart, rp2040_hal::gpio::PullDown>
        )
    > = uart::UartPeripheral::new(pac.UART0, uart_pins, &mut pac.RESETS)
        .enable(
            uart::UartConfig::new(
                115200.Hz(),
                uart::DataBits::Eight,
                None,
                uart::StopBits::One
            ),
            clocks.peripheral_clock.freq(),
        ).unwrap();

    let (rx, tx) = uart.split();

    let _ = tx.write_full_blocking(b"s");
    // we write s to start with so that the computer knows the pico is connect
    // and also so that we know that both motors are stationary.
    let mut rx_buf = [0u8; 1];
    loop {
        let _ = rx.read_raw(&mut rx_buf).expect("msg");
        let direction: &[u8; 1] = match rx_buf[0] {
            b'w' => {
                motor_one.forward();
                motor_two.forward();
                b"w"
            }
            b's' => {
                motor_one.stop();
                motor_two.stop();
                b"s"
            }
            b'a' => {
                motor_one.backwards();
                motor_two.forward();
                b"a"
            }
            b'd' => {
                motor_one.forward();
                motor_two.backwards();
                b"d"
            }
            b'x' => {
                motor_one.backwards();
                motor_two.backwards();
                b"x"
            }
            _ => {b"s"}
        };
    
    let _ = tx.write_raw(direction);
    led_pin.set_high().unwrap();
    }
}
