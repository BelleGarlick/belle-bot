import styled from "@emotion/styled";
import { THEME } from "../../Roboviz/utils.tsx";

export const Button = styled.button`
    padding: 12px 24px;
    background: ${THEME};
    border: 0px;
    border-radius: 6px;
    font-weight: bold;
    font-family: monospace;
    font-size: 16px;
    box-shadow: 0px 0px 10px 1000px rgba(0, 0, 0, 0) inset;
    transition: 0.1s all;

    :hover {
        box-shadow: 0px 0px 10px 1000px rgba(0, 0, 0, 0.15) inset;
    }
`;
