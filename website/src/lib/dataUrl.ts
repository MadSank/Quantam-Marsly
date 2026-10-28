/** URL of a file under public/data, relative to the deployed base. */
export const dataUrl = (rel: string) => `${import.meta.env.BASE_URL}data/${rel}`
