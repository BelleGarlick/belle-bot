import styled from "@emotion/styled";
import { THEME } from "../Roboviz/utils.tsx";

const HEIGHT = 50;
const LINK_WIDTH = 120;
const LOGO_WIDTH = 120;

export enum Page {
    REPLAYS = 0,
    DATASETS = 1,
    MODELS = 2,
    ROBOVIZ = 3,
}

const NavBarLink = styled.div`
    width: ${LINK_WIDTH}px;
    height: ${HEIGHT}px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    user-select: none;
    cursor: pointer;
`;

const Logo = styled.div`
    width: ${LOGO_WIDTH}px;
    height: ${HEIGHT}px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: ${THEME};
    font-weight: bold;
    user-select: none;
    text-shadow: 0px 0px 20px white;
`;

export function NavBar({
    page,
    setPage,
}: {
    page: Page;
    setPage: (page: Page) => void;
}) {
    return (
        <div
            style={{
                height: HEIGHT,
                display: "flex",
                flexDirection: "row",
                fontSize: 14,
                position: "relative",
                backgroundColor: "#111111",
            }}
        >
            <Logo>belle-bot</Logo>

            <NavBarLink onClick={() => setPage(Page.REPLAYS)}>
                replays
            </NavBarLink>

            <NavBarLink onClick={() => setPage(Page.MODELS)}>models</NavBarLink>

            <NavBarLink onClick={() => setPage(Page.DATASETS)}>
                datasets
            </NavBarLink>

            <NavBarLink onClick={() => setPage(Page.ROBOVIZ)}>
                roboviz
            </NavBarLink>

            <div
                style={{
                    position: "absolute",
                    top: HEIGHT - 2,
                    left: page * LINK_WIDTH + LOGO_WIDTH,
                    transition: "0.2s all ease",
                    width: LINK_WIDTH,
                    border: `1px solid ${THEME}`,
                    borderRadius: "10px 10px 0px 0px",
                }}
            />
        </div>
    );
}
