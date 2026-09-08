import { BottomBarSwitcher } from "./styles";
import { THEME } from "../../utils.tsx";

export interface SwitcherI {
    selection: string[];
    options: { text: string; value: string }[];
    setSelection: (selection: string[]) => void;
}

export function Switcher({ switcher }: { switcher: SwitcherI }) {
    return (
        <BottomBarSwitcher>
            {switcher.options.map((option) => {
                return (
                    <div
                        style={{
                            backgroundColor: switcher.selection.includes(
                                option.value,
                            )
                                ? THEME
                                : "black",
                            color: switcher.selection.includes(option.value)
                                ? "black"
                                : THEME,
                            padding: "0px 8px",
                        }}
                        onClick={() => {
                            switcher.setSelection([option.value]);
                        }}
                    >
                        {option.text}
                    </div>
                );
            })}
        </BottomBarSwitcher>
    );
}
