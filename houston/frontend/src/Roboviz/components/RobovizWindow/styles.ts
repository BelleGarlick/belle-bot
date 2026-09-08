import styled from "@emotion/styled";
import { THEME } from "../../utils.tsx";

export const TOPBAR_HEIGHT = 36;
export const BOTTOM_BAR_HEIGHT = 40;

export const WindowContainer = styled.div`
    position: absolute;
    outline: 2px solid #440088;
    color: ${THEME};
    overflow: hidden;
    background-color: #222;
    border-radius: 16px;
    box-shadow: 0px 0px 20px -10px black;
    user-select: none;
    box-sizing: border-box;
`;

export const WindowTopBar = styled.div`
    text-align: left;
    padding: 0px 12px;
    box-sizing: border-box;
    font-weight: bold;
    font-family: monospace;
    font-size: medium;
    cursor: grab;
    height: ${TOPBAR_HEIGHT}px;
    background-color: #222;
    user-select: none;
    display: flex;
    flex-direction: row;
    gap: 16px;
    align-items: center;
`;

export const WindowBottomBar = styled.div`
    text-align: left;
    box-sizing: border-box;
    padding: 0px 8px;
    font-weight: bold;
    font-family: monospace;
    font-size: small;
    height: ${BOTTOM_BAR_HEIGHT}px;
    background-color: #222;
    user-select: none;
    display: flex;
    flex-direction: row;
    gap: 8px;
    align-items: center;
`;

export const BottomBarSwitcher = styled.div`
    display: flex;
    flex-direction: row;
    border-radius: 6px;
    overflow: hidden;
    cursor: pointer;
    height: 26px;
`;

export const ResizeCorner = styled.div`
    position: absolute;
    bottom: 0;
    right: 0;
    width: 24px;
    height: 24px;
    cursor: nwse-resize;
    background-color: ${THEME};
    clip-path: polygon(100% 0, 100% 100%, 0 100%);
`;

export const DebugTextContainer = styled.div`
    position: absolute;
    left: 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
`;

export const DebugText = styled.div`
    background-color: #222;
    border-radius: 8px;
    padding: 0px 8px;
    font-family: monospace;
    font-size: 12px;
    box-shadow: 0px 0px 20px -10px black;
    font-weight: bold;
`;
