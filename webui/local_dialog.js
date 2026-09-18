export default {
  template: `
    <q-dialog :model-value="opened" @update:model-value="setOpen"
              @show="addClass" @hide="removeClass">
      <slot />
    </q-dialog>
  `,
  props: { modelValue: Boolean },
  emits: ["update:modelValue"],
  data() { return { opened: this.modelValue }; },
  watch: {
    modelValue(value) { this.opened = value; },
  },
  methods: {
    syncValue(value) { this.opened = value; },
    setOpen(value) {
      this.opened = value;
      this.$emit("update:modelValue", value);
    },
    addClass() { document.documentElement.classList.add("nicegui-dialog-open"); },
    removeClass() { document.documentElement.classList.remove("nicegui-dialog-open"); },
  },
  unmounted() { this.removeClass(); },
};
