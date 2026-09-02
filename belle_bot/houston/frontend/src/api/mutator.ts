export const customInstance = async <T>(
    url: string,
    options: RequestInit,
): Promise<T> => {
    const res = await fetch(url, options);

    if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
    }

    // Handle 204 No Content
    if (res.status === 204) {
        return {} as T;
    }

    const data = await res.json();
    return data as T;
};

export default customInstance;
