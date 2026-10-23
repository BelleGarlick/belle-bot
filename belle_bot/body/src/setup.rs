use rp2040_hal::clocks::ClocksManager;
use rp2040_hal::gpio::Pins;
use rp2040_hal::pac;
use rp2040_hal::sio::Sio;
use rp2040_hal::Watchdog;

pub fn setup_pins() -> (rp2040_hal::gpio::Pins, ClocksManager) {
    // TODO make it so that we parse pac through as an arg here or refactor as macro
    let mut pac: pac::Peripherals = pac::Peripherals::take().unwrap();
    // inialise a clock to keep track of timings
    let clocks: rp2040_hal::clocks::ClocksManager = rp2040_hal::clocks::init_clocks_and_plls(
        12_000_000u32,
        pac.XOSC,
        pac.CLOCKS,
        pac.PLL_SYS,
        pac.PLL_USB,
        &mut pac.RESETS,
        &mut Watchdog::new(pac.WATCHDOG), // prevents time out
    )
    .unwrap();

    let sio: Sio = Sio::new(pac.SIO);

    // initalise the pins, slice and uart_peripherals
    let pins: Pins = Pins::new(
        pac.IO_BANK0,
        pac.PADS_BANK0,
        sio.gpio_bank0,
        &mut pac.RESETS,
    );
    (pins, clocks)
}
