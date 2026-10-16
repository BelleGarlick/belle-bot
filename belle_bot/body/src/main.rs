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
///         Bat                DVR            PIC      LTP
///                [M1B]      
///         [+]    [M1A]       
///          |       |                       MICRO -> USBc
///          | ---> [+]            [DRV2] -> GPIO20
///          | ---> [-]____________[PWM2] -> GPIO21
///          |       |              [GRD] -> GPIO22
///         [-]    [M2A] ---->|         
///                [M1B] --]  | _____
///                        | [+]      \___
///                        ->[-]______/
///                           
///                             MTR
///
///
///
mod motor;

use defmt_rtt as _;
use panic_probe as _;

use embedded_hal::digital::OutputPin;
use rp2040_hal::pac;
use rp2040_hal::pwm;
use rp2040_hal::usb;
use rp2040_hal::{gpio::Pins, pwm::Slices, Sio, Watchdog};

use usb_device::bus::UsbBusAllocator;
use usb_device::prelude::*;
use usbd_serial::SerialPort;

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

    // sets up 
    let mut motor_one: motor::Motor<pwm::Pwm1> = motor::Motor::from(pins.gpio19, 19, pwm_slices.pwm1);
    let mut motor_two: motor::Motor<pwm::Pwm2> = motor::Motor::from(pins.gpio21, 21, pwm_slices.pwm2);

    // sets gpio25 onboard led to high so that we can see the pico has been flashed.
    let mut led_pin: rp2040_hal::gpio::Pin<
        rp2040_hal::gpio::bank0::Gpio25,
        rp2040_hal::gpio::FunctionSio<rp2040_hal::gpio::SioOutput>,
        rp2040_hal::gpio::PullDown,
    > = pins.gpio25.into_push_pull_output();

    // usb serial connection
    let usb_bus: UsbBusAllocator<usb::UsbBus> = UsbBusAllocator::new(usb::UsbBus::new(
        pac.USBCTRL_REGS,
        pac.USBCTRL_DPRAM,
        clocks.usb_clock,
        true,
        &mut pac.RESETS,
    ));

    // 3. Create the USB Serial (CDC-ACM) Port
    let mut serial = SerialPort::new(&usb_bus);

    // 4. Build the USB Device descriptor (Vendor ID & Product ID)
    let mut usb_dev = UsbDeviceBuilder::new(&usb_bus, UsbVidPid(0x2e8a, 0x000a))
        .strings(&[StringDescriptors::default()
            .manufacturer("Raspberry Pi")
            .product("Pico Motor Controller")
            .serial_number("12345678")])
        .unwrap()
        .device_class(usbd_serial::USB_CLASS_CDC)
        .build();

    loop {
        if usb_dev.poll(&mut [&mut serial]) {
            let mut buf = [0u8; 64];

            match serial.read(&mut buf) {
                Ok(count) if count > 0 => {
                    // Look at the incoming character sent from your laptop script/terminal
                    match buf[0] {
                        b'w' => {
                            motor_one.forward();
                            motor_two.forward();
                            let _ = serial.write(b"Motor moving STARTED\r\n");
                        }
                        b's' => {
                            motor_one.stop();
                            motor_two.stop();
                            let _ = serial.write(b"Motor STOPPED\r\n");
                        }
                        b'a' => {
                            motor_one.backwards();
                            motor_two.forward();
                            let _ = serial.write(b"Motor turning LEFT\r\n");
                        }
                        b'd' => {
                            motor_one.forward();
                            motor_two.backwards();
                            let _ = serial.write(b"Motor turning RIGHT\r\n");
                        }
                        b'x' => {
                            motor_one.backwards();
                            motor_two.backwards();
                            let _ = serial.write(b"Motor moving BACKWARD\r\n");
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        led_pin.set_high().unwrap();
    }
}
