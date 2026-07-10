import { fireEvent, render, screen } from "@testing-library/react";
import { ImageComparison, LanguageToggle } from "./components";
import i18n from "./i18n";

describe("portfolio interaction components", () => {
  it("moves the image comparison divider", () => {
    render(<ImageComparison before="before.png" after="after.png" />);
    const slider = screen.getByRole("slider", {
      name: "Before and after position",
    });
    fireEvent.change(slider, { target: { value: "72" } });
    expect(slider).toHaveValue("72");
    expect(screen.getByAltText("Restored").parentElement).toHaveStyle({
      clipPath: "inset(0 28% 0 0)",
    });
  });

  it("switches and persists the locale", async () => {
    await i18n.changeLanguage("en");
    render(<LanguageToggle />);
    fireEvent.click(screen.getByRole("button", { name: "Switch language" }));
    expect(i18n.language).toBe("zh");
    expect(localStorage.getItem("restorai.locale")).toBe("zh");
  });
});
